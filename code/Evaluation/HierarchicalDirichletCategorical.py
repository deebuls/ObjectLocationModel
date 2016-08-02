from bayespy import nodes
import numpy as np
import pandas as pd
from bayespy.inference import VB

class HierarchicalDirichletCategorical:
    '''
    Implements the Hierarchical Dirichlet Categorical model for the 
    object location model(OLM). 
    \alpha ~ 0.1
    p_conc ~ Concentration(<\alpha_1 .. \alpha_n >) where n = timezone to combine
    \theta ~ Dirichlet(p_conc)
    location ~ Categorical(\theta)
    '''
    
    def __init__(self, observed_locations):
        '''
        input : observed_locations
        '''
        self._observed_locations = observed_locations.copy()
        self._observed_locations.columns = ['time', 'location']
        
        #CONSTANTS
        #Maximum of the counts available Assumption that location is integer
        self.N_LOCATIONS = int(self._observed_locations['location'].unique().max() + 1)
        self.N_TIMEZONES = 24 #Discretized to hour basis
        self.N_OBSERVATIONS = pd.Series(np.zeros(self.N_TIMEZONES))
        self.N_OBSERVATIONS = self.N_OBSERVATIONS.add(self._observed_locations['time'].value_counts(), fill_value=0)
        print("Unique locations : ",self.N_LOCATIONS, type(self.N_LOCATIONS))
        #print("Number of Observations : ", self.N_OBSERVATIONS)
        
    def create_model(self, model_type=None):

        #Create location model for each of the timezone
        location_model = []

        if ('all' == model_type):
            p_conc = nodes.DirichletConcentration(self.N_LOCATIONS)
            p_conc.initialize_from_value(np.ones(self.N_LOCATIONS))
            p_theta = nodes.Dirichlet(p_conc,
                                      plates = (self.N_TIMEZONES,),
                                      name = 'p_theta')
            for time in np.arange(self.N_TIMEZONES):
                model = nodes.Categorical(p_theta[time],
                                        plates=(self.N_OBSERVATIONS[time],1),
                                        name=str(time))

                #observe data
                timezone_observations = self._observed_locations[self._observed_locations['time'] == time]

                if not timezone_observations.empty:
                    data = timezone_observations['location'].as_matrix().reshape((self.N_OBSERVATIONS[time],1))
                    model.observe(data)

                location_model.append(model)


            Q = VB(location_model[0], location_model[1], location_model[2], location_model[3],
                    location_model[4], location_model[5], location_model[6], location_model[7],
                    location_model[8], location_model[9], location_model[10], location_model[11],
                    location_model[12], location_model[13], location_model[14], location_model[15],
                    location_model[16], location_model[17], location_model[18], location_model[19],
                    location_model[20], location_model[21], location_model[22], location_model[23],
                    p_theta, p_conc)

        elif ('cross' == model_type):
            raise 'Not Implemented'
            pass
        elif ('2fold' == model_type):
            p_conc_morning = nodes.DirichletConcentration(self.N_LOCATIONS)
            p_conc_night = nodes.DirichletConcentration(self.N_LOCATIONS)

            p_conc_morning.initialize_from_value(np.ones(self.N_LOCATIONS))
            p_conc_night.initialize_from_value(np.ones(self.N_LOCATIONS))

            morning_time = np.arange(6,19)
            night_time = np.append(np.arange(0,6) , np.arange(18,24))

            p_theta_morning = nodes.Dirichlet(p_conc_morning,
                                      plates = (morning_time.size,),
                                      name = 'p_theta_morning')
            p_theta_night = nodes.Dirichlet(p_conc_night,
                                      plates = (night_time.size,),
                                      name = 'p_theta_night')


            #Combinging morning time
            for count, time in enumerate(morning_time):
                model = nodes.Categorical(p_theta_morning[count],
                                        plates=(self.N_OBSERVATIONS[time],1),
                                        name=str(time))

                #observe data
                timezone_observations = self._observed_locations[self._observed_locations['time'] == time]
                #print(timezone_observations)

                if not timezone_observations.empty:
                    data = timezone_observations['location'].as_matrix().reshape((self.N_OBSERVATIONS[time],1))
                    model.observe(data)

                location_model.append(model)

            #Combinging night time
            for count, time in enumerate(night_time):
                model = nodes.Categorical(p_theta_night[count],
                                        plates=(self.N_OBSERVATIONS[time],1),
                                        name=str(time))

                #observe data
                timezone_observations = self._observed_locations[self._observed_locations['time'] == time]

                if not timezone_observations.empty:
                    data = timezone_observations['location'].as_matrix().reshape((self.N_OBSERVATIONS[time],1))
                    model.observe(data)

                location_model.append(model)

            Q = VB(location_model[0], location_model[1], location_model[2], location_model[3],
                    location_model[4], location_model[5], location_model[6], location_model[7],
                    location_model[8], location_model[9], location_model[10], location_model[11],
                    location_model[12], location_model[13], location_model[14], location_model[15],
                    location_model[16], location_model[17], location_model[18], location_model[19],
                    location_model[20], location_model[21], location_model[22], location_model[23],
                    p_theta_morning, p_theta_night, p_conc_morning, p_conc_night)
        else:
            raise 'no model_type selected'

        print ("models created")

        ####################################################################################
        #Learning parameters
        Q.update(repeat=1000)
        print ('learned params')
        ####################################################################################
        
        if ('all' == model_type):
            return np.array(p_theta.get_parameters()).reshape((self.N_TIMEZONES,self.N_LOCATIONS))
        elif ('2fold' == model_type):
            learned_night = np.array(p_theta_night.get_parameters()).reshape((night_time.size, self.N_LOCATIONS))
            learned_morn = np.array(p_theta_morning.get_parameters()).reshape((morning_time.size, self.N_LOCATIONS))
            return(np.row_stack((learned_night[:6,:], learned_morn, learned_night[6:,:])))


