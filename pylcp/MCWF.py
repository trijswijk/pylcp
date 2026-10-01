"""
author: Lajos Palanki

MCWF method for pylcp.

"""
import numpy as np
from .integration_tools import solve_ivp_random
from .common import (progressBar, random_vector, cart2spherical, base_force_profile)
from .governingeq import governingeq
from copy import deepcopy
from numba import jit, types
from numba.typed import Dict
import copy
import time
 
def _abs2(x):
    return x.real**2 + x.imag**2

def _dagger(x):
    return np.einsum("...ij->...ji",x.conj())

def cartesian_vector_tensor_dot(a, B):
    if B.ndim == 2 and a.ndim == 1:
        # Single point:
        return np.dot(B, a)
    elif B.ndim == 2:
        # Constant B, variable a:
        return np.sum(a[np.newaxis, ...]*B[..., np.newaxis], axis=1)
    else:
        # Varaible a and variable B.  Will throw an error if a.shape[1:] != B.shape[2:]:
        return np.sum(a[np.newaxis, ...]*B[...], axis=1)

class force_profile(base_force_profile):
    """
    Monte-Carlo Wavefunction force profile

    The force profile object stores all of the calculated quantities created by
    the MCWF.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R  : array_like, shape (3, ...)
        Positions at which the force profile was calculated.
    V : array_like, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : array_like, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : array_like, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of array_like
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.laserBeams object used to create the governing equation.
    f_q : dictionary of array_like
        The force due to each laser and its :math:`q` component, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """
    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.iterations = np.zeros(self.R[0].shape, dtype='int64')
        self.fq = {}
        for key in laserBeams:
            self.fq[key] = np.zeros(self.R.shape + (3, len(laserBeams[key].beam_vector)))

    def store_data(self, ind, Neq, F, F_laser, F_mag, iterations, F_laser_q):
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for jj in range(3):
            for key in F_laser_q:
                self.fq[key][(jj,) + ind] = F_laser_q[key][jj]

        self.iterations[ind] = iterations

class MCWF(governingeq):
    
    """
    The Monte-Carlo Wavefunction method
    
    This class constructs the Monte-Carlo wavefunction solver for a given
    """
    
    def __init__(self, laserBeams, magField, hamiltonian,
                 a=np.array([0.,0.,0.]), r0=np.array([0.,0.,0.]),
                 v0=np.array([0.,0.,0.]), include_mag_forces=True):
        super().__init__(laserBeams, magField, hamiltonian, a, r0, v0)
        
        self.ev_mat = {}
        
        self.include_mag_forces = include_mag_forces
 
        self._build_coherent_ev()
        self._build_decay_ev()
        self.static_H = np.zeros((self.hamiltonian.n,self.hamiltonian.n),np.complex128)
        self.static_H += self.ev_mat["H0"]
        self.static_H += (-1j/2)*self.decay_ev
        
    def _build_coherent_ev(self):
        self.ev_mat["H0"] = self.hamiltonian.H_0
        self.ev_mat['B'] = self.hamiltonian.mu_q
        
        self.ev_mat['d_q'] = {}
        self.ev_mat["d_q*"] = {}
        for key in self.laserBeams.keys():
            self.ev_mat['d_q'][key] = self.hamiltonian.d_q_bare[key]
            self.ev_mat['d_q*'][key] = self.hamiltonian.d_q_star[key]
        
        # Helper arrays
        self.indicies = np.array(list(range(self.hamiltonian.n)))
        self.indicies2D = np.indices((self.hamiltonian.n,self.hamiltonian.n))
        self.flat_size = self.hamiltonian.n**2
        
        self.profile = {}
        self.sol = None
        
    
    def _build_decay_ev(self):
 
        self.decay_op = np.zeros((3,self.hamiltonian.n,self.hamiltonian.n),np.complex128)
        self.decay_ev = np.zeros((self.hamiltonian.n,self.hamiltonian.n),np.complex128)
        
        self.recoil_velocity = np.zeros((self.hamiltonian.n,self.hamiltonian.n))
        
        for key in self.hamiltonian.laser_keys:
            # The index of the current d_q matrix:
            ind = self.hamiltonian.laser_keys[key]
 
            # Grab the block of interest:
            d_q_block = self.hamiltonian.blocks[ind]
 
            # The offset index for the lower states:
            noff = int(np.sum(self.hamiltonian.ns[:ind[0]]))
            # The offset index for the higher states:
            moff = int(np.sum(self.hamiltonian.ns[:ind[1]]))
 
            # The number of lower states:
            n = self.hamiltonian.ns[ind[0]]
            # The number of higher states:
            m = self.hamiltonian.ns[ind[1]]
 
            # Calculate the decay of the states attached to this block:
            gamma = d_q_block.parameters['gamma']
            k = d_q_block.parameters['k']
 
            # Let's see if we can avoid a for loop here:
#             for jj in range(self.hamiltonian.ns[ll]):
#                 self.Rev_decay[n+jj, n+jj] -= gamma*\
#                             np.sum(np.sum(abs2(
#                                 other_block.matrix[:, :, jj]
#                             )))
            # # Save the decay rates out of the excited state:
            # self.decay_rates[key] = gamma*np.sum(_abs2(
            #     d_q_block.matrix[:, :, :]
            # ), axis=(0,1))
 
            # # # Save the indices for the excited states of this d_q block
            # # # for the random_recoil function:
            # # self.decay_N_indices[key] = np.arange(moff, moff+m)
 
            # # Add (more accurately, subtract) these decays to the evolution matrix:
            # self.decay_ev[(np.arange(moff, moff+m), np.arange(moff, moff+m))] -= \
            # self.decay_rates[key]
 
            # # Now calculate the decays into the lower state connected by this
            # # d_q:
            # self.decay_ev[noff:noff+n, moff:moff+m] += \
            #             gamma*np.sum(_abs2(d_q_block.matrix), axis=0)
 
            self.recoil_velocity[tuple(np.meshgrid(np.arange(noff, noff+n), np.arange(moff, moff+m),indexing="ij"))] = \
                k/self.hamiltonian.mass
 
            self.decay_op[tuple(np.meshgrid([0,1,2],np.arange(noff, noff+n), np.arange(moff, moff+m),indexing="ij"))] = \
               np.sqrt(gamma)*d_q_block.matrix
        
        self.decay_ev = np.einsum("qij,qjk->ik",_dagger(self.decay_op),self.decay_op)
        self.frac_decay = np.einsum("qij,jl,qji->il",_dagger(self.decay_op),np.eye(self.hamiltonian.n),self.decay_op)
        self.pol_decay = np.einsum("qij,qjk->qik",_dagger(self.decay_op),self.decay_op)
 
    def set_initial_psi(self, Psi0):
        """
        Sets the initial populations
 
        Parameters
        ----------
        Psi0 : array_like
            The initial state vector :math:`\Psi_0`.  It must have
            :math:`n` elements, where :math:`n` is the total number of states
            in the system.
        """
        if len(Psi0) != self.hamiltonian.n:
            raise ValueError('Npop has only %d entries for %d states.' %
                             (len(Psi0), self.hamiltonian.n))
        if np.any(np.isnan(Psi0)) or np.any(np.isinf(Psi0)):
            raise ValueError('Npop has NaNs or Infs!')
 
        self.Psi0 = np.asarray(Psi0,dtype=np.complex128)

    def set_initial_psi_equally(self):
        """
        Sets the initial populations
 
        Parameters
        ----------
        Psi0 : array_like
            The initial state vector :math:`\Psi_0`.  It must have
            :math:`n` elements, where :math:`n` is the total number of states
            in the system.
        """
 
        self.Psi0 = np.asarray(np.array([1/self.hamiltonian.n]*self.hamiltonian.n),dtype=np.complex128)

    def construct_evolution_matrix(self, r, v, t):
 
        # H  = self.ev_mat["H0"]
        n = self.hamiltonian.n
        H = np.zeros((n,n),np.complex128)
        H += self.static_H
        B = self.magField.Field(r, t)
        B = cart2spherical(B)
        if np.sum(np.abs(B))>1e-10:
            H -= np.einsum("q,qij->ij",[-1,1,-1]*B[::-1],self.ev_mat["B"])
 
        for key in self.laserBeams.keys():

            E = self.laserBeams[key].total_electric_field(np.real(r), t)
            # print(E)
            # H -= (np.einsum("q,qij->ij",[-1,1,-1]*E[::-1],self.ev_mat["d_q"][key]) +
            #         np.einsum("q,qij->ij",[-1,1,-1]*np.conjugate(E[::-1]),self.ev_mat["d_q*"][key]))/4
            
            # H -= np.sum(
            #       ([-1,1,-1]*E[::-1])[:,np.newaxis,np.newaxis]*self.ev_mat["d_q"][key] +
            #       ([-1,1,-1]*np.conjugate(E[::-1]))[:,np.newaxis,np.newaxis]*self.ev_mat["d_q*"][key],axis=0)/4
            
            
            H -= (np.dot([-1,1,-1]*E[::-1],self.ev_mat["d_q"][key].reshape((3,self.flat_size))) +
                  np.dot([-1,1,-1]*np.conjugate(E[::-1]),self.ev_mat["d_q*"][key].reshape((3,self.flat_size)))).reshape((self.hamiltonian.n,self.hamiltonian.n))/4
            
            # for jj, q in enumerate(np.arange(-1., 2., 1.)):
            #     self.H -= ((-1.)**q*E[2-jj]*self.ev_mat["d_q"][key][jj] +
            #                (-1.)**q*np.conjugate(E[2-jj])*self.ev_mat["d_q*"][key][jj])/4
            

            # self.H -= (np.tensordot(E[::-1]              ,([[[-1]],[[1]],[[-1]]] * self.ev_mat["d_q" ][key]), axes=(0,0)) + 
            #            np.tensordot(np.conjugate(E[::-1]),([[[-1]],[[1]],[[-1]]] * self.ev_mat["d_q*"][key]), axes=(0,0)))/4
        
        # Eq = {}
        # for key in self.laserBeams.keys():
        #     Eq[key] = self.laserBeams[key].total_electric_field(r, t)

        # B = self.magField.Field(r, t)
        # Bq = cart2spherical(B)

        # H = self.hamiltonian.return_full_H(Bq, Eq)
        
        # H += (-1j/2)*self.decay_ev
        
        self.Rev = -1j*H
        self.H = H
    
    # @staticmethod
    # @jit(nopython=True)
    # def bulk_observable(O, Psi):
    #     # return self.observable(O, Psi)
    #     return (np.einsum("li,...ij,lj->l...",np.conjugate(Psi),np.asarray(O,np.complex128),Psi))
    
    # @staticmethod
    # @jit(nopython=True)
    # def bulk_normalized_observable(O, Psi):
    #     # return self.normalized_observable(O, Psi)
    #     normed_Psi = Psi/np.sqrt(np.dot(np.conjugate(Psi),Psi))
    #     return (np.einsum("...i,qij,...j->...q",np.conjugate(normed_Psi),np.asarray(O,np.complex128),normed_Psi))
    
    @staticmethod
    # @jit(nopython=True)
    def observable(O, Psi):
        # return (np.einsum("i,...ij,j->...",np.conjugate(Psi),np.asarray(O,np.complex128),Psi))
        # return np.conjugate(Psi) @ np.asarray(O,np.complex128) @ Psi
        return np.squeeze(np.conjugate(Psi[...,np.newaxis,np.newaxis,:]) @ O[np.newaxis,...,:,:] @ Psi[...,np.newaxis,:,np.newaxis])

    @staticmethod
    # @jit(nopython=True)
    def normalized_observable(O, Psi):
        # normed_Psi = Psi/np.sqrt(np.dot(np.conjugate(Psi),Psi))
        normed_Psi = Psi/np.sqrt(np.sum(np.conjugate(Psi)*Psi,axis=-1)[...,np.newaxis])
        # return (np.einsum("i,...ij,j->...",np.conjugate(normed_Psi),np.asarray(O,np.complex128),normed_Psi))
        # return np.conjugate(normed_Psi) @ np.asarray(O,np.complex128) @ normed_Psi
        # return (np.conjugate(normed_Psi[...,np.newaxis,:]) @ O @ normed_Psi[...,:,np.newaxis])[...,0,0]
        return np.squeeze(np.conjugate(normed_Psi[...,np.newaxis,np.newaxis,:]) @ O[np.newaxis,...,:,:] @ normed_Psi[...,np.newaxis,:,np.newaxis])

    def force(self, r, t, Psi, return_details=False):
        # f = np.array([0.,0.,0.])
        f = np.zeros((3,) + Psi.shape[1:])
        if return_details:
            f_laser_q = {}
            f_laser = {}
        
        for key in self.laserBeams:
            # First, determine the average mu_q:
            # This returns a (3,) + rho.shape[2:] array
            d_q_av = self.normalized_observable(self.hamiltonian.d_q_bare[key], Psi.T)
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']

            if not return_details:

                delE = self.laserBeams[key].total_electric_field_gradient(np.real(r), t)
                # We are just looking at the d_q, whereas the full observable
                # is \nabla (d_q \cdot E^\dagger) + (d_q^* E)) =
                # 2 Re[\nabla (d_q\cdot E^\dagger)].  Putting in the units,
                # we see we need a factor of gamma/4, making
                # this 2 Re[\nabla (d_q\cdot E^\dagger)]/4 =
                # Re[\nabla (d_q\cdot E^\dagger)]/2
                # for jj, q in enumerate(np.arange(-1., 2., 1.)):
                #     f += np.real((-1)**q*gamma*d_q_av[jj]*delE[:, 2-jj])/2
                    
                # f += np.real(gamma*np.dot(delE[:,::-1],[-1,1,-1]*d_q_av))/2
                f += np.real(gamma*np.sum(delE[:,::-1]*(np.array([-1,1,-1]).reshape((1,3)+(1,)*(len(delE.shape)-2))*d_q_av.T[np.newaxis,:,...]),axis=1))/2
            else:
                f_laser_q[key] = np.zeros((3, 3, self.laserBeams[key].num_of_beams)
                            + Psi.shape[1:])
                f_laser[key] = np.zeros((3, self.laserBeams[key].num_of_beams)
                                        + Psi.shape[1:])

                # Now, dot it into each laser beam:
                for ii, beam in enumerate(self.laserBeams[key].beam_vector):
                    delE = beam.electric_field_gradient(r, t)
                    
                    for jj, q in enumerate(np.arange(-1., 2., 1.)):
                        f_laser_q[key][:, jj, ii] += \
                        np.real((-1)**q*gamma*d_q_av.T[jj]*delE[:, 2-jj])/2

                    f_laser[key][:, ii] = np.sum(f_laser_q[key][:, :, ii], axis=1)
                
                f+=np.sum(f_laser[key], axis=1)
                
                # Are we including magnetic forces?
        if self.include_mag_forces:
            # This function returns a matrix that is either (3, 3) (if constant)
            # or (3, 3, t.size).  The first two dimensions are like
            # [dBx/dx, dBy/dx, dBz/dx; dBx/dy, dBy/dy, dBz/dy], and so on.
            # We need to dot, and su
            delB = self.magField.gradField(np.real(r), t)

            # What's the expectation value of mu?  Returns (3,) or (3, t.size)
            av_mu = self.normalized_observable(self.hamiltonian.mu, Psi.T)

            # Now dot it into the gradient:
            f_mag = cartesian_vector_tensor_dot(av_mu, delB)

            # Add it into the regular force.
            f+=np.real(f_mag)
        elif return_details:
            f_mag=np.zeros(f.shape)

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f

    def evolve_state(self, t_span, max_scatter_probability = 0.1, rng=np.random.default_rng(),
                      progress_bar = False, debug = False, save_decay = False, decay_ops = "mF",
                      **kwargs):

 
 
        decay_events = []
 
        a = np.zeros((3,))
 
        if progress_bar:
            progress = progressBar()
 
        def dydt(t, y):
            r = y[-3:]
            v = y[-6:-3]
            Psi = y[:-6]
            self.construct_evolution_matrix(r,v,t)
            
            if progress_bar:
                progress.update(t/t_span[-1])
            
            return np.concatenate((
                # np.einsum("ij,j->i",self.Rev,Psi),
                self.Rev @ Psi,
                a,
                y[-6:-3]
                ))
 
        def random_func(t, y, dt):

            # Maybe introduces a bug. If step size is small enough then |Psi(t)> ~ |Psi(t+dt)> after renormalisation
            y[:-6] = y[:-6]/np.sqrt(np.sum(_abs2(y[:-6])))

            scat_amp = np.real(self.observable(self.decay_ev,y[:-6]))
            scat_rate = np.sum(scat_amp)
            scat_prob = dt*scat_rate
            dice = rng.random(1)[0]
            # print(scat_prob,end='\r')
            scatter = 1 if dice <= scat_prob else 0

            if (scatter):
                # decay_probs = dt*np.einsum("qi->i",np.real(self.observable(np.einsum("qij,jl,qjk->qlik",_dagger(self.decay_op),np.eye(self.hamiltonian.n),self.decay_op),y[:-6])))
                # decay_probs = dt*np.einsum("i,qij,jl,qji,i->il"
                #         ,np.conjugate(y[:-6]),_dagger(self.decay_op),np.eye(self.hamiltonian.n),self.decay_op,y[:-6])
                if decay_ops == 'mF':
                    decay_probs = dt*np.einsum("i,il->il",_abs2(y[:-6]),self.frac_decay)
                    (m,n) = self.indicies2D[:,(np.cumsum(decay_probs).reshape(decay_probs.shape)-dice)>=0][:,0]
                    if debug:
                        print(y[:-6],decay_probs/dt,n,dice,scat_prob/dt)
                    if save_decay:
                        decay_events.append([deepcopy(y[:-6]),decay_probs,n,dice,scat_prob,dt,t])
                    y[:-6] = np.zeros(self.hamiltonian.n)
                    y[n] = 1
                elif decay_ops == "pol":
                    decay_probs = dt*np.real(self.observable(self.pol_decay,y[:-6]))
                    n = np.array([0,1,2])[(np.cumsum(decay_probs).reshape(decay_probs.shape)-dice)>=0][0]
                    if debug:
                        print(y[:-6],decay_probs/dt,n-1,dice,scat_prob/dt)
                    if save_decay:
                        decay_events.append([deepcopy(y[:-6]),decay_probs,n-1,dice,scat_prob,dt,t])
                    y[:-6] = self.decay_op[n] @ y[:-6]
                    y[:-6] = y[:-6]/np.sqrt(np.sum(_abs2(y[:-6])))
                    
                scat_rate = np.sum(self.observable(self.decay_ev,y[:-6]))


            if scat_rate ==0:
                scat_rate = 1e-20

            max_timestep = max_scatter_probability/np.real(scat_rate)

            return scatter, np.real(max_timestep)
 
        y0 = np.concatenate((self.Psi0, self.v0, self.r0))
        init_scat_rate = np.sum(self.observable(self.decay_ev,y0[:-6]))
        if init_scat_rate ==0:
            init_scat_rate = 1e-20
        init_max_step = max_scatter_probability/np.real(init_scat_rate)
        
        sol = solve_ivp_random(dydt, random_func, t_span, y0, initial_max_step=init_max_step, **kwargs)
        sol.Psi = sol.y[:-6]/np.sqrt(self.observable(np.eye(self.hamiltonian.n),sol.y[:-6].T).T)
        sol.v = np.real(sol.y[-6:-3])
        sol.r = np.real(sol.y[-3:])
        if save_decay:
            sol.decay_events = decay_events
        self.sol = sol
        return sol


    def evolve_motion(self, t_span, max_scatter_probability = 0.1, rng=np.random.default_rng(),
                      progress_bar = False, debug = False, save_decay = False,
                      random_recoil = False, freeze_axis = [False,False,False], decay_ops = "mF",
                      **kwargs):
 
        """
        Evolve :math:`\\Psi_i` and the motion of the atom in time

        This method evolves the non-Hermitian hamiltonian and moves
        the particle along via the instantaneous force, for some
        period of time. It tracks decays and 

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        freeze_axis : list of boolean
            Freeze atomic motion along the specified axis.
            Default: [False, False, False]
        random_recoil : boolean
            Allow the atom to randomly recoil from scattering events. Without
            this scattering events are still tracked, but random recoil is not
            added as a result from these.
            Default: False
        max_scatter_probability : float
            When undergoing random recoils, this sets the maximum time step such
            that the maximum scattering probability is less than or equal to
            this number during the next time step.  Default: 0.1
        progress_bar : boolean
            If true, show a progress bar as the calculation proceeds.
            Default: False
        record_force : boolean
            If true, record the instantaneous force and store in the solution.
            Default: False
        save_decay : boolean
            If true records the individual decay events. It saves Psi at the time
            of the decay, the decay probabilities, the final state after the decay,
            the random dice roll, the total scattering probability, and time of the
            scatter.
        rng : numpy.random.Generator()
            A properly-seeded random number generator.  Default: calls
            ``numpy.random.default.rng()``
        **kwargs :
            Additional keyword arguments get passed to solve_ivp_random, which
            is what actually does the integration.

        Returns
        -------
        sol : OdeSolution
            Bunch object that contains the following fields:

                * t: integration times found by solve_ivp
                * Psi: density matrix
                * v: atomic velocity
                * r: atomic position
                * decay_events : The collection of decay properties. Only
                    present if save_decay is True

            It contains other important elements, which can be discerned from
            scipy's solve_ivp documentation.
        """
 
        decay_events = []
 
        free_axes = np.bitwise_not(freeze_axis)
 
        if progress_bar:
            progress = progressBar()
 
        def dydt(t, y):
            r = y[-3:]
            v = y[-6:-3]
            Psi = y[:-6]
            self.construct_evolution_matrix(r,v,t)
            F = self.force(r,t,Psi)
            # print(r)
            # F=np.array([0,0,0])
            if progress_bar:
                progress.update(t/t_span[-1])
            
            return np.concatenate((
                #np.einsum("ij,j->i",self.Rev,Psi),
                self.Rev @ Psi,
                free_axes*F/self.hamiltonian.mass + self.constant_accel,
                y[-6:-3]
                ))
 
        def random_func(t, y, dt):
            r = y[-3:]
            v = y[-6:-3]

            # Maybe introduces a bug. If step size is small enough then |Psi(t)> ~ |Psi(t+dt)> after renormalisation
            y[:-6] = y[:-6]/np.sqrt(np.sum(_abs2(y[:-6])))

            scat_amp = np.real(self.observable(self.decay_ev,y[:-6]))
            scat_rate = np.sum(scat_amp)
            scat_prob = dt*scat_rate
            dice = rng.random(1)[0]
            # print(scat_prob,end='\r')
            scatter = 1 if dice <= scat_prob else 0

            if (scatter):
                
                
                if decay_ops == 'mF':
                    decay_probs = dt*np.einsum("i,il->il",_abs2(y[:-6]),self.frac_decay)
                    (m,n) = self.indicies2D[:,(np.cumsum(decay_probs).reshape(decay_probs.shape)-dice)>=0][:,0]
                    if debug:
                        print(y[:-6],decay_probs/dt,n,dice,scat_prob/dt)
                    if save_decay:
                        decay_events.append([deepcopy(y[:-6]),decay_probs,n,dice,scat_prob,dt,t])
                    if random_recoil:
                        # Add random absorption + emission
                        # Does not account for the fact when the absorption does not come on the same transition!
                        # Does not account for cases where molasses are not used or there is a strong preference
                        # of absorption from a specific source!
                        y[-6:-3] += self.recoil_velocity[n,m]*(random_vector(rng, free_axes)+
                                                            random_vector(rng, free_axes))
                    y[:-6] = np.zeros(self.hamiltonian.n)
                    y[n] = 1
                elif decay_ops == "pol":
                    decay_probs = dt*np.real(self.observable(self.pol_decay,y[:-6]))
                    n = np.array([0,1,2])[(np.cumsum(decay_probs).reshape(decay_probs.shape)-dice)>=0][0]
                    if debug:
                        print(y[:-6],decay_probs/dt,n-1,dice,scat_prob/dt)
                    if save_decay:
                        decay_events.append([deepcopy(y[:-6]),decay_probs,n-1,dice,scat_prob,dt,t])
                    if random_recoil:
                        decay_probs = dt*np.einsum("i,il->il",_abs2(y[:-6]),self.frac_decay)
                        (mm,nn) = self.indicies2D[:,(np.cumsum(decay_probs).reshape(decay_probs.shape)-dice)>=0][:,0]
                        # Add random absorption + emission
                        # Does not account for the fact when the absorption does not come on the same transition!
                        # Does not account for cases where molasses are not used or there is a strong preference
                        # of absorption from a specific source!
                        y[-6:-3] += self.recoil_velocity[nn,mm]*(random_vector(rng, free_axes)+
                                                                 random_vector(rng, free_axes))
                    y[:-6] = self.decay_op[n] @ y[:-6]
                    y[:-6] = y[:-6]/np.sqrt(np.sum(_abs2(y[:-6])))
                
                scat_rate = np.sum(self.observable(self.decay_ev,y[:-6]))

            if scat_rate ==0:
                scat_rate = 1e-20

            max_timestep = max_scatter_probability/np.real(scat_rate)

            return scatter, np.real(max_timestep)

        y0 = np.concatenate((self.Psi0, self.v0, self.r0))
        init_scat_rate = np.sum(self.observable(self.decay_ev,y0[:-6]))
        if init_scat_rate ==0:
            init_scat_rate = 1e-20
        init_max_step = max_scatter_probability/np.real(init_scat_rate)
        
        sol = solve_ivp_random(dydt, random_func, t_span, y0, initial_max_step=init_max_step, **kwargs)
        sol.Psi = sol.y[:-6]/np.sqrt(self.observable(np.eye(self.hamiltonian.n),sol.y[:-6].T).T)
        sol.v = np.real(sol.y[-6:-3])
        sol.r = np.real(sol.y[-3:])
        if save_decay:
            sol.decay_events = decay_events
        self.sol = sol
        return sol
    
    
    def find_equilibrium_force(self, deltat=500, itermax=100, Npts=5001,
                               rel=1e-5, abs=1e-9, debug=False,
                               initial_psi ='equally',
                               return_details=False, **kwargs):
        """
        Finds the equilibrium force at the initial position

        This method works by solving evolving the state for a chunk of time
        :math:`\\Delta T`, calculating the force during that chunck, continuing
        the integration for another chunck, calculating the force during that
        subsequent chunck, and comparing the average of the forces of the two
        chunks to see if they have converged.

        Parameters
        ----------
        deltat : float
            Chunk time :math:`\\Delta T`.  Default: 500
        itermax : int, optional
            Maximum number of iterations.  Default: 100
        Npts : int, optional
            Number of points to divide the chunk into.  Default: 5001
        rel : float, optional
            Relative convergence parameter.  Default: 1e-5
        abs : float, optional
            Absolute convergence parameter.  Default: 1e-9
        debug : boolean, optional
            If true, pring out debug information as it goes.
        initial_rho : 'rateeq' or 'equally'
            Determines how to set the initial rho at the start of the
            calculation.
        return_details : boolean, optional
            If true, returns the forces from each laser and the scattering rate
            matrix.

        Returns
        -------
        F : array_like
            total equilibrium force experienced by the atom
        F_laser : dictionary of array_like
            If return_details is True, the forces due to each laser, indexed
            by the manifold the laser addresses.  The dictionary is keyed by
            the transition driven, and individual lasers are in the same order
            as in the pylcp.laserBeams object used to create the governing
            equation.
        F_laser : dictionary of array_like
            If return_details is True, the forces due to each laser and its q
            component, indexed by the manifold the laser addresses.  The
            dictionary is keyed by the transition driven, and individual lasers
            are in the same order as in the pylcp.laserBeams object used to
            create the governing equation.
        F_mag : array_like
            If return_details is True, the forces due to the magnetic field.
        Neq : array_like
            If return_details is True, the equilibrium populations.
        ii : int
            Number of iterations needed to converge.
        """
        # if initial_rho == 'rateeq':
        #     self.set_initial_rho_from_rateeq()
        if initial_psi == 'equally':
            self.set_initial_psi_equally()
        elif initial_psi == 'frompops':
            Npop = kwargs.pop('init_pop', None)
            self.set_initial_psi(Npop)
        else:
            raise ValueError('Argument initial_psi=%s not understood'%initial_psi)

        old_f_avg = np.array([np.inf, np.inf, np.inf])

        if debug:
            print('Finding equilbrium force at '+
                  'r=(%.2f, %.2f, %.2f) ' % (self.r0[0], self.r0[1], self.r0[2]) +
                  'v=(%.2f, %.2f, %.2f) ' % (self.v0[0], self.v0[1], self.v0[2]) +
                  'with deltat = %.2f, itermax = %d, Npts = %d, ' %  (deltat, itermax, Npts) +
                  'rel = %.1e and abs = %.1e' % (rel, abs)
                  )
            self.piecewise_sols = []

        ii=0
        while True:
            if not Npts is None:
                kwargs['t_eval'] = np.linspace(ii*deltat, (ii+1)*deltat, int(Npts))

            self.evolve_state([ii*deltat, (ii+1)*deltat], **kwargs)
            f, f_laser, f_laser_q, f_mag = self.force(self.sol.r, self.sol.t, self.sol.Psi,
                                                      return_details=True)

            f_avg = np.mean(f, axis=1)

            if debug:
                print(ii, f_avg, np.sum(f_avg**2))
                self.piecewise_sols.append(self.sol)

            if (np.sum((f_avg)**2)<abs or
                np.sum((old_f_avg-f_avg)**2)/np.sum((f_avg)**2)<rel or
                np.sum((old_f_avg-f_avg)**2)<abs):
                break;
            elif ii>=itermax-1:
                break;
            else:
                old_f_avg = copy.copy(f_avg)
                self.set_initial_psi(self.sol.Psi[:, -1])
                self.set_initial_position_and_velocity(self.sol.r[:, -1],
                                                       self.sol.v[:, -1])
                ii+=1

        if return_details:
            f_mag = np.mean(f_mag, axis=1)

            f_laser_avg = {}
            f_laser_avg_q = {}
            for key in f_laser:
                f_laser_avg[key] = np.mean(f_laser[key], axis=2)
                f_laser_avg_q[key] = np.mean(f_laser_q[key], axis=3)

            Neq = np.real(np.mean(np.conjugate(self.sol.Psi)*self.sol.Psi, axis=1))
            return (f_avg, f_laser_avg, f_laser_avg_q, f_mag, Neq, ii)
        else:
            return f_avg


    def generate_force_profile(self, R, V, name=None, progress_bar=False,
                               **kwargs):
        """
        Map out the equilibrium force vs. position and velocity

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector.  First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            repsectively.
        V : array_like, shape(3, ...)
            Velocity vector.  First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
            repsectively.
        name : str, optional
            Name for the profile.  Stored in profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.
        progress_bar : boolean, optional
            Displays a progress bar as the proceeds.  Default: False

        Returns
        -------
        profile : pylcp.obe.force_profile
            Resulting force profile.
        """
        def default_deltat(r, v, deltat_v, deltat_r, deltat_tmax):
            deltat = None
            if deltat_v is not None:
                vabs = np.sqrt(np.sum(v**2))
                if vabs==0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2*np.pi*deltat_v/vabs, deltat_tmax])

            if deltat_r is not None:
                rabs = np.sqrt(np.sum(r**2))
                if rabs==0.:
                    deltat = deltat_tmax
                else:
                    deltat = np.min([2*np.pi*deltat_r/rabs, deltat_tmax])

            return deltat

        deltat_r = kwargs.pop('deltat_r', None)
        deltat_v = kwargs.pop('deltat_v', None)
        deltat_tmax = kwargs.pop('deltat_tmax', np.inf)
        deltat_func = kwargs.pop(
            'deltat_func',
            lambda r, v: default_deltat(r, v, deltat_v, deltat_r, deltat_tmax)
        )

        if not name:
            name = '{0:d}'.format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

        it = np.nditer([R[0], R[1], R[2], V[0], V[1], V[2]],
                       flags=['refs_ok', 'multi_index'],
                        op_flags=[['readonly'], ['readonly'], ['readonly'],
                                  ['readonly'], ['readonly'], ['readonly']])

        if progress_bar:
            progress = progressBar()

        for (x, y, z, vx, vy, vz) in it:
            # Construct the rate equations:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            if progress_bar:
                tic = time.time()

            self.set_initial_position_and_velocity(r, v)

            if not deltat_func(r, v) is None:
                kwargs['deltat'] = deltat_func(r, v)
            kwargs['return_details'] = True

            F, F_laser, F_laser_q, F_mag, Neq, iterations = self.find_equilibrium_force(**kwargs)

            self.profile[name].store_data(it.multi_index, Neq, F, F_laser, F_mag,
                                          iterations, F_laser_q)

            if progress_bar:
                progress.update((it.iterindex+1)/it.itersize)

        return self.profile[name]
