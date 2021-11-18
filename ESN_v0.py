import numpy as np
from scipy import linalg,stats
from sklearn.linear_model import Ridge,LinearRegression
import warnings

NoneType = type(None)

sigmoid = lambda k: 1 / (1 + np.exp(-k))

mse = lambda k,l: np.square(k-l).mean()

mape = lambda k,l: np.abs(1-l/k).mean()

class EchoStateNetwork:

    """
    DOCUMENTATION
    -

    Author: Ege Yilmaz
    Year: 2021
    Title: Echo State Network class for master's thesis.

        - TRAIN: Excite the reservoir states and train the linear readout via regression.

            - Regular: With inputs. Either Teacher Forced or Input Driven.

                - Teacher Forced: Uses past version of the data to be fit during training.
                x(n+1) = (1-alpha) * x(n) + alpha*f(Win * u(n+1) + W * x(n) + Wback * y(n))

                - Input Driven: Not teacher forced. Just inputs.
                x(n+1) = (1-alpha) * x(n) + alpha*f(Win * u(n+1) + W * x(n))

            - Output Feedback: No inputs, reservoir and outputs fed back to the reservoir. Either Teacher Forced or Autonomous.
            
                - Teacher Forced: Feedback is past versions of the data to be predicted.
                x(n+1) = (1-alpha) * x(n) + alpha*f(W * x(n) + Wback * y(n))

            - Self Feedback: No inputs, no outputs. Reservoir dynamics depends on itself.
                x(n+1) = (1-alpha) * x(n) + alpha*f(W * x(n)).


        - PREDICT: Use validation input data and/or output data, which will be fed back to the reservoir as input, to do forecasts.
        Can be Generative, Predictive or with Output Feedback.

            - Regular: With Inputs.

                - Generative: Reservoir states are updated by using reservoir's outputs.
                    x(n+1) = (1-alpha) * x(n) + alpha*f(Win * u(n+1) + W * x(n) + Wback * Wout * (1;u(n);y_predicted(n-1);x(n)))

                - Predictive: Reservoir states are updated by using past versions of the data to be predicted.
                    x(n+1) = (1-alpha) * x(n) + alpha*f(Win * u(n+1) + W * x(n) + Wback * y(n))

                - Input Driven: Inputs are the only external contributers to the dynamics of the reservoir.
                    x(n+1) = (1-alpha) * x(n) + alpha*f(Win * u(n+1) + W * x(n) )
                    
            - Output Feedback: No inputs, outputs are fed back to the reservoir.

                - Teacher Forced: Feedback is the past version of the data to be predicted.
                    x(n+1) = (1-alpha) * x(n) + alpha*f(W * x(n) + Wback * y(n))

                - Autonomous: Feedback is reservoir's predictions of the past version of the data to be predicted.
                    x(n+1) = (1-alpha) * x(n) + alpha*f(W * x(n) + Wback * Wout * (1;y_predicted(n-1);x(n)))

        - TEST: Additional mode for testing training success. Reservoir is initialized from zero state with the help of teacher output for a given period 
        and then left running freely.

    """

    def __init__(self,
                W: np.ndarray=None,
                resSize: int=400,
                xn: list=[0,4,-4],
                pn: list=[0.9875, 0.00625, 0.00625],
                random_state: float=None,
                **kwargs):
        
        """ 

        Description
        -
        Generates the reservoir matrix: Initializes the units and their connections.

        Variables
        -

            - W: User can provide custom reservoir matrix

            - resSize: Number of units in the reservoir.

            - xn , pn: User can provide custom random variable that generates sparse reservoir matrix.
            xn are the values and pn are the corresponding probabilities.

            - random_state: Fix random state.

            - keyword agruments:
                
                - verbose: Mute all message outputs.
        
        """
        
        assert W is None or (len(W.shape)==2 and W.shape[0]==W.shape[1] and isinstance(W,np.ndarray))
        assert isinstance(resSize,int), "Please give integer reservoir size."

        self.resSize = resSize if W is None else W.shape[0]
        self._inSize = None
        self._outSize = None

        self.W = 0.1 * stats.rv_discrete(name='sparse', \
                        values=(xn, pn)).rvs(size=(resSize,resSize) \
                                ,random_state=random_state) if W is None else W

        self._X = None
        self._X_val = None
        self.Win = None
        self.Wout = None
        self.Wback = None
        self.states = None
        self.val_states = None
        self._update_rule_id = None
        self.f = None
        self.f_out = None
        self.f_out_inverse = None
        self.output_transformer = None

        if random_state:
            np.random.seed(int(random_state))

        self.spectral_radius = abs(linalg.eig(self.W)[0]).max()
        if kwargs.get("verbose",1):
            print(f'Reservoir generated. Spectral Radius: {self.spectral_radius}')



    def scale_reservoir_weights(self,desired_spectral_radius: float):

        """ Scales the reservoir matrix to have the desired spectral radius."""

        assert isinstance(desired_spectral_radius,float)

        print(f'Scaling matrix to have spectral radius {desired_spectral_radius}...')
        self.W *= desired_spectral_radius / self.spectral_radius
        self.spectral_radius = abs(linalg.eig(self.W)[0]).max()
        print(f'Done: {self.spectral_radius}')
        


    def excite(self,
                u: np.ndarray=None,
                y: np.ndarray=None,
                bias: int=True,
                f=sigmoid,
                alpha: float=0.3,
                initLen: int=None, 
                trainLen: int=None,
                initTrainLen_ratio: float=None,
                wobble: bool=False,
                wobbler: np.ndarray=None,
                null_state_init: bool=True,
                custom_initState: np.ndarray=None,
                **kwargs) -> NoneType:
        """

        Description
        -
        Stimulate reservoir states either with given inputs and/or outputs or let it excite itself without input and output.

        Variables
        -

            - u: Input. Has shape [...,time].

            - y: To be predicted. Has shape [...,time].

            - bias: enables bias in the input, reservoir and readout connections.

            - f: User can provide custom activation function. Default is sigmoid activation.

            - alpha: leaking rate in x(n) = (1-alpha)*x(n-1) + alpha*f(...) .Default 0.3

            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
            Will be set to an eighth of the training length if not provided.

            - trainLen: Total no of training steps. Will be set to the length of input data.

            - initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. 
            An input of 8 would mean that the initialization period will be an eighth of the training length.

            - wobble: For enabling random noise.

            - wobbler: User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - null_state_init: If True, starts the reservoir from null state. If False, starts them randomly. Default is True.

            - custom_initState: User can give custom initial reservoir state x(0).

            - keyword arguments:

                - Win: custom input weights
        """

        # Some stuff needs checking right out the bat.

        validation_mode = kwargs.get("validation_mode",False)
        assert bool(initLen)+bool(initTrainLen_ratio) < 2, "Please give either initLen or initTrainLen_ratio."
        assert isinstance(u,(np.ndarray,NoneType)) and isinstance(y,(np.ndarray,NoneType)), f'Please give numpy arrays. type(u):{type(u)} and type(y):{type(y)}'
        
        # Update rule recognition based on function inputs

        update_rule_id = kwargs.get("update_rule_id",None)
        if update_rule_id is None:
            update_rule_id = (bool(u is not None) + 1)*(bool(u is not None)+bool(y is not None))
            update_rule_id = 3 if update_rule_id == 4 else update_rule_id
        training_type_dict = {0:"Self Feedback",1:"Output feedback/Teacher forced",2:"Regular/Input driven",3:"Regular/Teacher forced"}
        
        """
        0: both no
        1: no u yes y
        2: yes u no y
        3: both yes
        """

        #Handling I/O

        if not validation_mode:

            self.training_type = training_type_dict[update_rule_id]

            if update_rule_id % 2 - 1: #if y is None
                assert wobbler is None and not wobble ,"Wobble states are desired only in the case of teacher forced setting."
            
            if update_rule_id > 1: #if u is not None:
                # u = u.copy()
                assert len(u.shape) == 2
                inSize = u.shape[0]
                trainLen = u.shape[-1] if trainLen is None else trainLen
                """ Input and Feedback Connections """
                self.Win = kwargs.get("Win",None)
                if self.Win is None:
                    self.Win = np.random.rand(self.resSize,bias+inSize) - 0.5
                    # Win = np.random.uniform(size=(self.resSize,inSize+bias))<0.5
                    # self.Win = np.where(Win==0, -1, Win)

            if update_rule_id % 2:  #if y is not None:
                y_ = y
                assert len(y.shape) == 2
                outSize = y.shape[0]
                trainLen = y.shape[-1] if trainLen is None else trainLen

                
                """ Input and Feedback Connections """
                self.Wback = np.random.uniform(-2,2,size=(self.resSize,outSize)) if kwargs.get("Wback",self.Wback) is None else kwargs.get("Wback",self.Wback)

                """Wobbler"""
                assert isinstance(wobble,bool),"wobble parameter must be boolean."
                if wobbler is not None:
                    assert y.shape == wobbler, "Wobbler must have shape same as the output."
                    self._wobbler = wobbler
                    y_ = y + self._wobbler
                elif wobble:
                    self._wobbler = np.random.uniform(-1,1,size=y.shape)/10000
                    y_ = y + self._wobbler
                else:
                    self._wobbler = 0

        else:
            inSize = self._inSize
            outSize = self._outSize



        # Initialization and Training Lengths
        assert initTrainLen_ratio is None or initTrainLen_ratio >= 1, "initTrainLen_ratio must be larger equal than 1."
        if initLen is None:
            initLen = trainLen//initTrainLen_ratio if initTrainLen_ratio else trainLen//8
        assert initLen >= 1 or validation_mode
        self.initLen = initLen


        # Bias
        bias = int(bias)
        self.bias = bias

        # Exciting the reservoir states
        assert trainLen is not None
        if custom_initState is None:
            x = np.zeros((self.resSize,)) if null_state_init else np.random.rand(self.resSize,)
        else:
            assert custom_initState.shape == (self.resSize,),f"Please give custom initial state with shape ({self.resSize},)."
            x = custom_initState

        # no u, no y
        if update_rule_id == 0:
            assert isinstance(trainLen,int), f"Training length must be integer.{trainLen} is given."
            X = np.zeros((bias+self.resSize,trainLen-initLen))
            if validation_mode:
                x = self.states[:,-1]
                if self._update_rule_id == 1:
                    # training was with no u, yes y. now validation with no u, yes y_pred
                    X = np.zeros((bias+outSize+self.resSize,trainLen-initLen))
                    y_temp = self.output_transformer(self.f_out(np.dot(self.Wout, self._X[:,-1])))
                    for t in range(trainLen):
                        x = (1-alpha)*x + alpha*f(np.dot( self.W, x ) + np.dot( self.Wback, y_temp ))
                        X[:,t] =  np.concatenate((bool(bias)*[bias],y_temp,x)).ravel()
                        y_temp = self.output_transformer(self.f_out(np.dot(self.Wout, X[:,t])) + self._wobbler[:,t])
                    states = X[bias+outSize:,:]

                elif self._update_rule_id == 2:
                    # training was with yes u, no y. now validation with yes u_pred, no y
                    # This is only useful when input data and output data differ by a phase.
                    X = np.zeros((bias+inSize+self.resSize,trainLen-initLen))
                    u_temp = self.output_transformer(self.f_out(np.dot(self.Wout, self._X[:,-1])))
                    for t in range(trainLen):
                        x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u_temp)).T) + np.dot( self.W, x ))
                        X[:,t] = np.concatenate((bool(bias)*[bias],u_temp,x)).ravel()
                        u_temp = self.output_transformer(self.f_out(np.dot(self.Wout, X[:,t])))
                    states = X[inSize+bias:,:] 

                else:
                    # training was with no u, no y
                    for t in range(trainLen):
                        x = (1-alpha)*x + alpha*f(np.dot( self.W, x ))
                        X[:,t] = np.concatenate((bool(bias)*[bias],x)).ravel()
                    states = X[bias:,:]
            else:
                for t in range(1,trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot( self.W, x ))
                    if t >= initLen:
                        X[:,t-initLen] = np.concatenate((bool(bias)*[bias],x)).ravel()
                states = X[bias:,:]
        # no u, yes y
        elif update_rule_id == 1:
            X = np.zeros((bias+outSize+self.resSize,trainLen-initLen))
            if validation_mode:
                # no u, yes y
                x = self.states[:,-1]
                y_temp = self._y_train_last
                for t in range(trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot( self.W, x ) + np.dot(self.Wback, y_temp))
                    X[:,t-initLen] = np.concatenate((bool(bias)*[bias],y_temp,x)).ravel()
                    y_temp = y[:,t]  + self._wobbler[:,t]
            else:
                for t in range(1,trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot( self.W, x ) + np.dot(self.Wback, y_[:,t-1]))
                    if t >= initLen:
                        X[:,t-initLen] = np.concatenate((bool(bias)*[bias],y_[:,t-1],x)).ravel()
                self._outSize = outSize
                self._y_train_last = y_[:,-1]
            states = X[outSize+bias:,:]
        # yes u, no y
        elif update_rule_id == 2:
            X = np.zeros((bias+inSize+self.resSize,trainLen-initLen))
            if validation_mode:
                x = self.states[:,-1]
                if self._update_rule_id == 3:
                    # yes u, yes y_pred (generative)
                    X = np.zeros((bias+inSize+outSize+self.resSize,trainLen-initLen))
                    y_temp = self.output_transformer(self.f_out(np.dot(self.Wout, self._X[:,-1])))
                    for t in range(trainLen):
                        x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u[:,t])).T) \
                                    + np.dot( self.W, x ) + np.dot(self.Wback, y_temp))
                        X[:,t] = np.concatenate((bool(bias)*[bias],u[:,t],y_temp,x)).ravel()
                        y_temp = self.output_transformer(self.f_out(np.dot(self.Wout, X[:,t])) + self._wobbler[:,t])
                    states = X[bias+inSize+outSize:,:]
                else:
                    # yes u, no y
                    for t in range(trainLen):
                        x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u[:,t])).T) + np.dot( self.W, x ))
                        X[:,t] = np.concatenate((bool(bias)*[bias],u[:,t],x)).ravel()
                    states = X[inSize+bias:,:]
            else:
                for t in range(1,trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u[:,t])).T) + np.dot( self.W, x ))
                    if t >= initLen:
                        X[:,t-initLen] = np.concatenate((bool(bias)*[bias],u[:,t],x)).ravel()
                self._inSize = inSize
                states = X[inSize+bias:,:] 
        # yes u, yes y
        elif update_rule_id == 3:
            assert u.shape[-1] == y.shape[-1], "Inputs and outputs must have same shape at the last axis (time axis)."
            X = np.zeros((bias+inSize+outSize+self.resSize,trainLen-initLen))
            if validation_mode:
                x = self.states[:,-1]
                y_temp = self._y_train_last
                for t in range(trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u[:,t])).T) \
                            + np.dot( self.W, x ) + np.dot(self.Wback, y_temp))
                    X[:,t] = np.concatenate((bool(bias)*[bias],u[:,t],y_temp,x)).ravel()
                    y_temp = y[:,t] + self._wobbler[:,t]
            else:
                for t in range(1,trainLen):
                    x = (1-alpha)*x + alpha*f(np.dot(self.Win, np.concatenate((bool(bias)*[bias],u[:,t])).T) \
                                 + np.dot( self.W, x ) + np.dot(self.Wback, y_[:,t-1]))
                    if t >= initLen:
                        X[:,t-initLen] = np.concatenate((bool(bias)*[bias],u[:,t],y_[:,t-1],x)).ravel()
                self._inSize = inSize
                self._outSize = outSize
                self._y_train_last = y_[:,-1]
            states = X[inSize+outSize+bias:,:]
        #?
        else:
            raise NotImplementedError("Could not find a case for this training.")       

        
        assert states.shape[0] == self.resSize
        if validation_mode:
            self.val_states = states
            self._X_val = X
        else:
            self._update_rule_id = update_rule_id
            self._X = X
            self.states = states
            self._bias = bias
            self.alpha = alpha
            self.f = f



    def train(self,
                y: np.ndarray,
                f_out_inverse=None,
                regr=None,
                reg_type: str="ridge",
                ridge_param: float=1e-8,
                solver: str="auto",
                error_measure: str="mse",
                **kwargs) -> np.ndarray:

        """ 
        
        Description
        -

        Trains the readout via linear or ridge regression from scikit-learn:
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        
        Returns the error of selected type.

        Variables
        -

        - y: Data to fit.

        - f_out_inverse: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.

        - regr: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.

        - reg_type: Regression type. Can be ridge or linear. Default is linear.

        - ridge_param: Regularization factor in ridge regression.

        - solver: See scikit documentation.

        - error_measure: Can be 'mse' or 'mape'.

        - keyword arguments:

                - verbose: For the error message.

        """

        assert isinstance(y,np.ndarray), f'Please give numpy array. type(y):{type(y)}'

        if regr is None:
            if reg_type.lower() == "ridge":
                regr = Ridge(ridge_param, fit_intercept=False,solver=solver)
            else:
                regr = LinearRegression(fit_intercept=False)
        if f_out_inverse is not None:
            self.f_out_inverse = f_out_inverse
            y_ = f_out_inverse(y)
        else:
            y_ = y

        regr.fit(self._X.T,y_.T)
        self.Wout = regr.coef_

        bias = self.Wout.shape[-1] - self.resSize
        if self._inSize is not None:
            bias -= self._inSize
        if self._outSize is not None:
            bias -= self._outSize

        assert bias == 1 or bias == 0, bias

        if error_measure == "mse":
            error = mse(y_,np.dot( self.Wout , self._X))
            self.mse_train = error
        elif error_measure == "mape":
            error = mape(y_,np.dot( self.Wout , self._X))
            self.mape_train = error
        else:
            raise NotImplementedError("Unknown error measure type.")
        
        if kwargs.get("verbose",1):
            print(error_measure.upper(),": ",error)

        return error

        

    def validate(self,
                u: np.ndarray=None,
                y: np.ndarray=None,
                valLen: int=None,
                f_out=lambda x:x,
                output_transformer=lambda x:x,
                **kwargs) -> np.ndarray:

        """

        -VARIABLES-

        u: input

        y: to be predicted

        - valLen:  Training length. If u or y is provided it is not needed to be set. Mostly necessary for when neither u nor y is present.

        - f_out: Custom output activation. Default is identity.

        - output_transformer: Transforms the reservoir outputs at the very end. Default is identity.
        
        - keyword arguments:

            - bias: Enables bias in the input, reservoir and readout connections. Default is the one used in training

            - f: User can provide custom reservoir activation function. Default is the one used in training.

            - alpha: Leaking rate in x(n) = (1-alpha)*x(n-1) + alpha*f(...) .Default is the alpha used in training.

            - wobble: For enabling random noise. Default is False.

            - wobbler: User can provide custom noise. Disabled per default.

        """

        assert self.Wout is not None
        assert isinstance(u,(np.ndarray,NoneType)) and isinstance(y,(np.ndarray,NoneType)), f'Please give numpy arrays. type(u):{type(u)} and type(y):{type(y)}'


        # Update rule recognition based on function inputs
        update_rule_id = (bool(u is not None) + 1)*(bool(u is not None)+bool(y is not None))
        update_rule_id = 3 if update_rule_id == 4 else update_rule_id
        assert u is not None or y is not None or valLen is not None
        

        # Bias
        bias = kwargs.get("bias",self._bias)
        self._val_bias = bias

        # Activation
        f = kwargs.get("f",self.f)
        self.val_f = f
        self.f_out = f_out
        self.output_transformer = output_transformer

        # Leaking Rate
        alpha = kwargs.get("alpha",self.alpha)
        self.val_alpha = alpha


        validation_rule_dict = {
                                0:
                                    {0:"Self-Feedback"},
                                1:
                                    {0:"Output Feedback/Autonomous", 1:"Output Feedback/Teacher Forced"},
                                2:
                                    {2:"Regular/Input Driven",0:"Input Feedback"},
                                3:
                                    {2:"Regular/Generative",3:"Regular/Predictive"}
                                }

        self.val_type = validation_rule_dict[self._update_rule_id][update_rule_id]

        if self.val_type == validation_rule_dict[0][0]:
            warnings.warn(f"You are forecasting in {validation_rule_dict} mode!")

        if (self._update_rule_id - update_rule_id) == 1 and update_rule_id%2 == 0:
            pass
        elif self._update_rule_id == 2 and update_rule_id==0:
            pass
        else:
            assert self._update_rule_id == update_rule_id \
                ,f"You trained the network in {self.training_type} mode but trying to forecast in {self.val_type} mode."
        
        if u is not None:
            assert self._inSize == u.shape[0], "Please give input consistent with training input."
        if y is not None:
            assert self._outSize == y.shape[0], "Please give output consistent with training output."
        if self._bias != int(bias):
            warnings.warn(f"You have used {self._bias} during training but now you are using {int(bias)}.")
        if self.f != f:
            warnings.warn(f"You have used {self.f} reservoir activation during training but now you are using {f}.")
        if self.alpha != alpha:
            warnings.warn(f"You have used leaking rate {self.alpha} during training but now you are using {alpha}.")

        if u is not None:
            valLen = u.shape[-1]
        elif y is not None:
            valLen = y.shape[-1]
        else:
            valLen=valLen

        # Wobbler
        wobble = kwargs.get("wobble",False)
        wobbler = kwargs.get("wobbler",None)
        assert self._update_rule_id % 2 or not wobble
        assert wobbler is None or wobble
        if wobble and wobbler is None:
           self._wobbler = np.random.uniform(-1,1,size=(self.Wout.shape[0],valLen))/10000
        elif wobbler is not None:
            self._wobbler = wobbler
        else:
            self._wobbler = np.zeros(shape=(self.Wout.shape[0],valLen))

        self.excite(u, y, bias, f, alpha, initLen=0,trainLen=valLen,wobble=wobble,wobbler=self._wobbler,update_rule_id=update_rule_id,validation_mode=True)

        return self.output_transformer(self.f_out(np.dot(self.Wout, self._X_val)))



    def session(self,
                X_t: np.ndarray=None,
                y_t: np.ndarray=None,
                X_v: np.ndarray=None,
                y_v: np.ndarray=None,
                training_data: np.ndarray=None,
                bias: int=True,
                f=sigmoid,
                f_out_inverse=None,
                f_out=lambda x:x,
                output_transformer=lambda x:x,
                alpha: float=0.3,
                initLen: int=None, 
                initTrainLen_ratio: float=None,
                trainLen: int=None,
                valLen: int=None,
                wobble_train: bool=False,
                wobbler_train: np.ndarray=None,
                null_state_init: bool=True,
                custom_initState: np.ndarray=None,
                regr=None,
                reg_type: str="ridge",
                ridge_param: float=1e-8,
                solver: str="auto",
                error_measure: str="mse",
                **kwargs
                ) -> np.ndarray:
        
        """

        Description
        -
        Executes the class methods excite, train and validate. Returns predictions.

        Variables
        -

            - X_t: Training inputs. Has shape [...,time].

            - y_t: Training targets. Has shape [...,time].

            - X_v: Validation inputs. Has shape [...,time].

            - y_v: Validation targets. Has shape [...,time].

            - training_data: Data to be fit. It will be set to y_t automatically if it is not provided.

            - bias: Enables bias in the input, reservoir and readout connections.

            - f: User can provide custom reservoir activation function. Default is sigmoid activation.
            
            - f_out_inverse: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.
            
            - f_out: Custom output activation. Default is identity.

            - output_transformer: Transforms the reservoir outputs at the very end. Default is identity.

            - alpha: Leaking rate in x(n) = (1-alpha)*x(n-1) + alpha*f(...) . Default 0.3

            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
            Will be set to an eighth of the training length if not provided.

            - initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. 
            An input of 8 would mean that the initialization period will be an eighth of the training length.

            - trainLen: Total no of training steps. Will be set to the length of input data.

            - valLen: Total no of validation steps. Will be set to the length of input data.

            - wobble_train: For enabling random noise.

            - wobbler_train: User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - null_state_init: If True, starts the reservoir from null state. If False, starts them randomly. Default is True.

            - custom_initState: User can give custom initial reservoir state x(0).

            - keyword arguments:

                - Win: Custom input weights.

                - Wback: Custom output feedback weights.

                - bias_val: Enables bias in the input, reservoir and readout connections. Default is bias used in training.

                - f_val: User can provide custom reservoir activation function. Default is activation used in training.

                - alpha_val: Leaking rate in x(n) = (1-alpha)*x(n-1) + alpha*f(...) . Default is alpha used in training.
                
                - wobble_val: For enabling random noise. Default is False.
                
                - wobbler_val: User can provide custom noise. Disabled per default.

                - train_only: Set to True to perform a training session only, i.e. no validation is done.

                - verbose: For the training error messages.


        """
        assert y_t is not None or training_data is not None

        training_data = y_t if y_t is not None else training_data

        self._X = None
        self._X_val = None
        self.Win = None
        self.Wout = None
        self.Wback = None
        self.states = None
        self.val_states = None
        self._update_rule_id = None
        self.f = None
        self.f_out = None
        self.f_out_inverse = None
        self.output_transformer = None

        self.excite(u=X_t,
                    y=y_t,
                    bias=bias,
                    f=f,
                    alpha=alpha,
                    initLen=initLen,
                    trainLen=trainLen,
                    initTrainLen_ratio=initTrainLen_ratio,
                    wobble=wobble_train,
                    wobbler=wobbler_train,
                    null_state_init=null_state_init,
                    custom_initState=custom_initState,
                    Win=kwargs.get("Win",None),
                    Wback=kwargs.get("Wback",None)
                    )
        
        self.train(y=training_data[:,self.initLen:],
                    f_out_inverse=f_out_inverse,
                    regr=regr,
                    reg_type=reg_type,
                    ridge_param=ridge_param,
                    solver=solver,
                    error_measure=error_measure,
                    verbose=kwargs.get("verbose",1)
                    )

        if kwargs.get("train_only"):
            return np.dot( self.Wout , self._X)

        pred = self.validate(u=X_v,
                    y=y_v,
                    valLen=valLen,
                    f_out=f_out,
                    output_transformer=output_transformer,
                    bias=kwargs.get("bias_val",self._bias),
                    f=kwargs.get("f_val",self.f),
                    alpha=kwargs.get("alpha_val",self.alpha),
                    wobble = kwargs.get("wobble_val",False),
                    wobbler = kwargs.get("wobbler_val",None)
                    )
        
        return pred



    def test():
        "TBD"
        pass
