


from typing import Union
import numpy as np
import torch
import torch.nn as nn



class Constraint(nn.Module):

    def __init__(self, constraint_name:str) -> None:

        super().__init__()
        self.constraint_name = constraint_name


    def constraint_function(self, const_parameters:Union[nn.ParameterDict, nn.Parameter, torch.Tensor]):
        """
        The constraint function.
        """
        raise NotImplementedError
    
    def evaluate_constraint(self, const_parameters:Union[nn.ParameterDict, nn.Parameter, torch.Tensor]):
        """
        Evaluates the constraint function.

        If the constraint defined by `constraint function` is satisfied, then its evaluation is 0. 
        Else, it is a positive number (follows the flag logic in `_flag_constraint()`)
        """
        return self.constraint_function(const_parameters)
    
    def update_psi(self, psi:Union[nn.ParameterDict, nn.Parameter, torch.Tensor], 
                         theta: Union[nn.ParameterDict, nn.Parameter, torch.Tensor], 
                         lag_multiplier: Union[nn.ParameterDict, nn.Parameter, torch.Tensor]):
        """
        Updates the \psi values of the ADMM algothm.
        """
        raise NotImplementedError
    
    def _flag_constraint(self, x):
        """
        If the constraint is not satisfied, i.e., if Constraint < 0, then we flag it with the relu function. 
        This also enforces the non-negativity constraint.
        """
        return torch.relu(-x)
    
    def __str__(self) -> str:
        return self.constraint_name


class Nonnegativity_Constraint(Constraint):
    
        def __init__(self) -> None:
            super().__init__('nonnegativity')

        def evaluate_constraint(self, const_parameters:nn.ParameterDict):
            """
            Evaluates the constraint function.

            If `const_parameters` is a dict, then the constraint function is evaluated for each item in the dict.
            """

            if isinstance(const_parameters, dict):
                return {key: self._flag_constraint(value) for key, value in const_parameters.items()}
            
            raise TypeError('The constraint parameters must be either a nn.ParameterDict or a nn.Parameter.')

        def update_psi(self, psi:nn.ParameterDict, theta: nn.ParameterDict, lag_multiplier: nn.ParameterDict):
            """
            Updates the \psi values of the ADMM algothm.

            Nonnegativity constraint is applied to all parameters in \psi

            Returns
            -------
            psi : nn.ParameterDict
                The updated \psi values.
            """
            
            eval = self.evaluate_constraint(psi)

            psi_n_plus_1 = {}
            
            for key in eval:
                if eval[key] == 0: # If the constraint is satisfied, then we update the \psi value.
                    psi_n_plus_1[key] = torch.tensor(theta[key] + lag_multiplier[key], device=psi[key].device, dtype=psi[key].dtype)
                    #print(f'psi[{key}] = {psi[key]} = {theta[key]} + {lag_multiplier[key]}')
                else: 
                    # if the constraint is not satisfied, that means that the \psi value is non-negative, 
                    # thus we project it to the feasible set, i.e., to 0.
                    psi_n_plus_1[key] = torch.tensor(0.0, device=psi[key].device, dtype=psi[key].dtype)

            return psi_n_plus_1


class Arrow_Constraint(Constraint):
    """
    This constraint is used to ensure that the cone radius is greater than the radius of the base cylinder.
    This way, we ensure that the arrow mainstains its geometry and does not become a cylinder or a pencil.
    """

    def __init__(self) -> None:
        super().__init__('arrow')

    def constraint_function(self, const_parameters:dict):
        """
        Returns a dictionary with the constraint values for the parameters of interest of this contraint 
        (i.e., the cone radius of each arrow GENEO).
        """

        geneo_names = [key for key in const_parameters.keys() if 'cone_' in key and 'geneo' in key]
        geneo_names = np.array([key.split('.')[-3] for key in geneo_names])
        geneo_names = np.unique(geneo_names)

        return {f'geneos.{geneo}.geneo_params.cone_radius': 
                self._flag_constraint(const_parameters[f'geneos.{geneo}.geneo_params.cone_radius'] - 2*const_parameters[f'geneos.{geneo}.geneo_params.radius']) 
                for geneo in geneo_names}

        #return {'cone_radius': self._flag_constraint(const_parameters['cone_radius'] - 2*const_parameters['radius'])}
    

    def evaluate_constraint(self, const_parameters:dict):
            """
            Evaluates the constraint function.
            """

            if isinstance(const_parameters, dict):
                return self.constraint_function(const_parameters)
            
            raise TypeError('The constraint parameters must be a nn.ParameterDict.')


    def update_psi(self, psi:dict, theta:dict, lag_multiplier: dict):
            
            if isinstance(psi, dict) and isinstance(theta, dict) and isinstance(lag_multiplier, dict):
                eval = self.evaluate_constraint(psi)
                psi_n_plus_1 = {}
                for key in eval:
                    if eval[key] == 0: # If the constraint is satisfied, then we update the \psi value.
                        psi_n_plus_1[key] = theta[key] + lag_multiplier[key]
                    else: 
                        # if the constraint is not satisfied, that means that the \psi value is non-negative, 
                        # thus we project it to the feasible set.
                        #psi_n_plus_1[key] = torch.tensor(0.0, device=psi[key].device, dtype=psi[key].dtype)
                        psi_n_plus_1[key] = torch.tensor(2*psi[f'geneos.{key.split(".")[-3]}.geneo_params.radius'], device=psi[key].device, dtype=psi[key].dtype)
                return psi_n_plus_1
            else:
                raise TypeError(f'The parameters must ALL be dictionaries. Instead, we have: \n psi: {type(psi)} \n theta: {type(theta)} \n lag_multiplier: {type(lag_multiplier)}')
    

class Cylinder_Constraint(Constraint):
    """
    This constraint is used to ensure that the cylinder height is greater than the 2*radius of the cylinder.
    This way, we aim to maintain a narrow cilindric shape that resembles more a tower.
    """

    def __init__(self, kernel_height) -> None:
        super().__init__('cylinder')

        self.kernel_height = kernel_height

    def constraint_function(self, const_parameters:nn.ParameterDict, kernel_height):

        geneo_names = [key for key in const_parameters.keys() if 'cy_' in key and 'geneo' in key]
        geneo_names = np.array([key.split('.')[-3] for key in geneo_names])
        geneo_names = np.unique(geneo_names)

        return {f"geneos.{geneo}.geneo_params.radius": 
                self._flag_constraint(kernel_height/2 - const_parameters[f"geneos.{geneo}.geneo_params.radius"]) for geneo in geneo_names}

    
    def evaluate_constraint(self, const_parameters:nn.ParameterDict):
        return self.constraint_function(const_parameters, self.kernel_height)

    def update_psi(self, psi:nn.ParameterDict, theta: nn.ParameterDict, lag_multiplier: nn.ParameterDict):
        
            if isinstance(psi, dict) and isinstance(theta, dict) and isinstance(lag_multiplier, dict):
                eval = self.evaluate_constraint(psi)
                psi_n_plus_1 = {}
                for key in eval:
                    if eval[key] == 0: # If the constraint is satisfied, then we update the \psi value.
                        psi_n_plus_1[key] = theta[key] + lag_multiplier[key]
                    else: 
                        # if the constraint is not satisfied, that means that the \psi value is non-negative, 
                        # thus we project it to the feasible set.
                        psi_n_plus_1[key] = torch.tensor(self.kernel_height/2, device=psi[key].device, dtype=psi[key].dtype) #torch.tensor(self.kernel_height/2)
                return psi_n_plus_1 
            else:
                raise TypeError(f'The parameters must ALL be Dictionaries. Instead, we have: \n psi: {type(psi)} \n theta: {type(theta)} \n lag_multiplier: {type(lag_multiplier)}')
