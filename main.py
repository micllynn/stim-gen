"""
STIMULUS GENERATOR

Created on Tue Sep  5 10:42:19 2017

@author: Emerson


Class with built-in methods for generating commonly used stimuli and writing
them to ATF files for use with AxonInstruments hardware.

Example usage:
    
    # Initialize the class and simulate a synaptic current.
    s = Stim('Slow EPSC')
    s.generate_PS(duration = 200, ampli = 10, tau_rise = 1.5, tau_decay = 15)
    
    # Display some information about the generated waveform.
    print(s)
    s.plot()
    
    # Create a set of synaptic-like currents of increasing amplitude.
    s.set_replicates(5)
    s.command *= np.arange(1, 6)
    s.plot()
    
    # Write the stimulus to an ATF file.
    s.write_ATF()
"""




#%% IMPORT PREREQUISITE MODULES

import numpy as np
import numba as nb
import matplotlib.pyplot as plt




#%% DEFINE MAIN STIM CLASS

class Stim(object):
    
    """
    Class with built-in methods for generating commonly used stimuli and writing them to ATF files for use with AxonInstruments hardware.
    
    Attributes:
        
        label           -- string descriptor of the class instance
        stim_type       -- string descriptor of the type of stimulus.
        dt              -- size of the time step in ms.
        command         -- 2D array containing stimuli; time across rows, sweeps across cols.
        time            -- time support vector.
    
    
    Methods:
        
        generate_PS         -- generate a synaptic current/potential-like waveform.
        generate_OU         -- generate Ornstein-Uhlenbeck noise.
        set_replicates      -- set the number of replicates of the stimulus.
        plot                -- plot the stimulus.
        write_ATF           -- write the stimulus to an ATF file.
    
    
    Example usage:
        
        # Initialize the class and simulate a synaptic current.
        s = Stim('Slow EPSC')
        s.generate_PS(duration = 200, ampli = 10, tau_rise = 1.5, tau_decay = 15)
        
        # Display some information about the generated waveform.
        print(s)
        s.plot()
        
        # Create a set of synaptic-like currents of increasing amplitude.
        s.set_replicates(5)
        s.command *= np.arange(1, 6)
        s.plot()
        
        # Write the stimulus to an ATF file.
        s.write_ATF()
    """
    
    
    ### MAGIC METHODS
    
    # Initialize class instance.
    def __init__(self, label, dt = 0.1):
        
        self.label      = label
        self.stim_type  = 'Empty'
        
        self.dt         = dt        # Sampling interval in ms.
        
        self.command    = None      # Attribute to hold the command (only current is currently supported).
        self.time       = None      # Attribute to hold a time support vector.
        
    
    # Method for unambiguous representation of Stim instance.
    def __repr__(self):
        
        if not self.time is None:
            time_range  = '[' + str(self.time[0]) + ', ' + str(self.time[-1]) + ']'
            command_str   = np.array2string(self.command)
        else:
            time_range  = str(self.time)
            command_str   = str(self.command)
        
        return 'Stim object\n\nLabel: ' + self.label + '\nStim type: ' + self.stim_type + '\nTime range (ms): ' + time_range + '\nTime step (ms):' + str(self.dt) + '\nCommand:\n' + command_str 
    
    
    # Pretty print self.command and some important details.
    # (Called by print().)
    def __str__(self):
        
        header = self.stim_type + ' Stim object'
        
        # Include more details about the object if it isn't empty.
        if not self.command is None:
            header += ' with ' + str( self.command.shape[1] ) + ' sweeps of ' + str( ( self.time[-1] + self.dt ) / 1000 ) + 's each.\n\n'
            content = np.array2string(self.command)
        
        else:
            content = ''
        
        return str(self.label) + '\n\n' + header + content
            
    
    
    ### MAIN METHODS
    
    # Generate a synaptic current-like waveform.
    def generate_PS(self, duration, ampli, tau_rise, tau_decay):
        
        """
        Generate a post-synaptic potential/current-like waveform.
        
        Note that the rise and decay time constants are only good approximations of fitted rise/decay taus (which are more experimentally relevant) if the provided values are separated by at least approx. half an order of magnitude.
        
        Inputs:
            duration        -- length of the simulated waveform in ms ^ -1.
            ampli           -- peak height of the waveform.
            tau_rise        -- time constant of the rising phase of the waveform in ms ^ -1.
            tau_decay       -- time constant of the falling phase of the waveform in ms ^ -1.
        """
        
        # Initialize time support vector.
        offset = 500
        self.time = np.arange(0, duration, self.dt)
        
        # Generate waveform based on time constants then normalize amplitude.
        waveform = np.abs( np.exp( -self.time / tau_rise ) - np.exp( -self.time / tau_decay ) )
        waveform /= np.max(waveform)
        waveform *= ampli
        
        # Convert waveform into a column vector.
        waveform = np.concatenate( ( np.zeros( (int( offset / self.dt)) ), waveform ), axis = 0 )
        waveform = waveform[np.newaxis].T
        
        # Assign output.
        self.time = np.arange(0, duration + offset, self.dt)
        self.command    = waveform
        self.stim_type  = "Post-synaptic current-like"
    
    
    # Realize OU noise and assign to self.command. (Wrapper for _gen_OU_internal.)
    def generate_OU(self, duration, I0, tau, sigma0, dsigma, sin_per, bias_factor, bias_point):
        
        """
        Realize Ornstein-Uhlenbeck noise.
        
        Parameters are provided to allow the noise SD to vary sinusoidally over time.
        
        sigma[t] = sigma0 * ( 1 + dsigma * sin(2pi * sin_freq)[t] )
        
        Additionally, the noise distribution may be skewed according to an exponential function. Positive values of bias_factor produce an upwards skew, and zero eliminates skew.
        
        Inputs:
            duration        -- duration of noise to realize in ms.
            I0              -- mean value of the noise.
            tau             -- noise time constant in ms ^ -1.
            sigma0          -- mean SD of the noise.
            dsigma          -- fractional permutation of noise SD.
            sin_per         -- period of the sinusoidal SD permutation in ms.
            bias_factor     -- slope of the upward bias function.
            bias_point      -- point at which the value of the bias function is 1.
        """
            
        
        # Initialize support vectors.
        self.time       = np.arange(0, duration, self.dt)
        self.command    = np.zeros(self.time.shape)
        S               = sigma0 * ( 1 + dsigma * np.sin( (2 * np.pi / sin_per ) * self.time))
        rands           = np.random.standard_normal( len(self.time) )
        
        # Perform type conversions for vectors.
        self.time.dtype         = np.float64
        self.command.dtype      = np.float64
        S.dtype                 = np.float64
        rands.dtype             = np.float64
        
        # Perform type conversions for constants.
        self.dt                 = np.float64( self.dt )
        I0                      = np.float64( I0 )
        tau                     = np.float64( tau )
        bias_factor             = np.float64( bias_factor )
        bias_point              = np.float64( bias_point )
        
        # Realize noise using nb.jit-accelerated function.
        noise = self._gen_OU_internal(self.time, 
                                      rands, 
                                      self.dt, 
                                      I0,
                                      tau,
                                      S,
                                      bias_factor,
                                      bias_point)
        
        # Convert noise to a column vector.
        noise = noise[np.newaxis].T
        
        # Assign output.
        self.command    = noise
        self.stim_type  = 'Ornstein-Uhlenbeck noise'
        
    
    # Generate sinusoidal input
    def generate_sin(self, duration, I0, ampli, period):
        
        """
        Generate a sine wave with time-dependent amplitude and/or period.
        
        Inputs:
            duration        -- duration of the wave in ms.
            I0              -- offset of the wave.
            ampli           -- amplitude of the wave.
            period          -- period of the wave in ms.
            
        Amplitude and/or period can be time-varied by passing one-dimensional vectors of length duration/dt instead of constants.
        """
        
        # Initialize time support vector.
        self.time = np.arange(0, duration, self.dt)
        
        # Convert ampli to a vector if need be;
        # otherwise check that it's the right shape.
        try:
            tmp = iter(ampli); del tmp
            assert len(ampli) == len(self.time)
            
        except TypeError:
            ampli = np.array( [ampli] * len(self.time) )
            
        except AssertionError:
            raise ValueError('len of ampli must correspond to duration.')
        
        # Do the same with period.
        try:
            tmp = iter(period); del tmp
            assert len(period) == len(self.time)
        
        except TypeError:
            period = np.array( [period] * len(self.time) )
            
        except AssertionError:
            raise ValueError('len of period must correspond to duration.')
        
        # Calculate the sine wave over time.
        sinewave = I0 + ampli * np.sin( ( 2 * np.pi / period ) * self.time )
        
        # Convert sine wave to column vector.
        sinewave = sinewave[np.newaxis].T
        
        # Assign output.
        self.command    = sinewave
        self.stim_type  = 'Sine wave'
        
    
    # Set number of replicates of the command array.
    def set_replicates(self, reps):
        
        """
        Set number of replicates of the existing command array.
        """
        
        # Check that command has been initialized.
        try:
            assert not ( self.command   is None )
        except AssertionError:
            raise RuntimeError('No command array to replicate!')
        
        # Create replicates by tiling.
        self.command = np.tile(self.command, (1, reps))
    
    
    # Plot command, time, and additional data.
    def plot(self, **data):
        
        """
        Plot command (and any additional data) over time.
        
        Produces a plot of self.command over self.time as its primary output. 
        
        Additional data of interest may be plotted as supplementary plots by passing them to the function as named arguments each containing a numerical vector of the same length as self.command.
        """
        
        d_keys  = data.keys()
        l_dk    = len(d_keys)
        
        plt.figure(figsize = (9, 3 + 3 * l_dk))
        plt.suptitle( str(self.label) )
        
        # Plot generated noise over time.
        plt.subplot(1 + l_dk, 1, 1)
        plt.title('Generated stimulus')
        plt.xlabel('Time (ms)')
        plt.ylabel('Command')
        
        plt.plot(self.time, self.command, '-k', linewidth = 0.5)
        
        # Add plots from data passed as named arguments.
        i = 2
        for key in d_keys:
            
            plt.subplot(1 + l_dk, 1, i)
            plt.title( key )
            plt.xlabel('Time (ms)')
            
            plt.plot(self.time, data[ key ], '-k', linewidth = 0.5)
            
            i += 1
        
        
        # Final formatting and show plot.
        plt.tight_layout(rect = (0, 0, 1, 0.95))
        plt.show()
          
            
    # Write command and time to an ATF file.
    def write_ATF(self):
        
        """
        Write command and time to an ATF file in the current working directory.
        """
        
        # Check whether there is any data to write.
        try:
            assert not ( self.command   is None )
            assert not ( self.time      is None )
        except AssertionError:
            raise RuntimeError('Command and time must both exist!')
        
        fname = self.label + '.ATF'
        header = ''.join(['ATF1.0\n1\t{0}\nType=1\nTime (ms)\t'.format(self.command.shape[1]),
                          *['Command (AU)\t' for sweep in range(self.command.shape[1])], '\n'])
        
        # Convert numeric arrays to strings.
        str_command     = self.command.astype( np.unicode_ )
        str_time        = self.time.astype( np.unicode_ )
        
        # Initialize list to hold arrays.
        contentl = []
        
        # Tab-delimit data one row (i.e., time step) at a time.
        for t in range( len(str_time) ):
            
            tmp = str_time[t] + '\t' + '\t'.join( str_command[t, :] )
            contentl.append(tmp)
        
        # Turn the content list into one long string.
        content = '\n'.join(contentl)
        
        # Write the header and content strings to the file.
        with open(fname, 'w') as f:
            
            f.write(header)
            f.write(content)
            f.close()
    
    
    
    ### HIDDEN METHODS
    
    # Fast internal method to realize OU noise. (Called by generate_OU.)
    @staticmethod
    @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64), nopython = True)
    def _gen_OU_internal(T, rands, dt, I0, tau, sigma, bias_factor, bias_point):
        
        I       = np.zeros(T.shape, dtype = np.float64)
        I[0]    = I0
        
        for t in range(1, len(T)):
            
            adaptive_term = dt * ( ( I0 - I[t - 1] ) + bias_factor * np.exp(-(I[t - 1] - bias_point) / 5) )/ tau
            random_term = np.sqrt( 2 * dt * sigma[t]**2 / tau ) * rands[t]
            
            I[t] = I[t - 1] + adaptive_term + random_term
        
        return I