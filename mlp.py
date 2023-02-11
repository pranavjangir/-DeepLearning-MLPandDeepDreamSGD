import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        linearop1 = torch.matmul(x, torch.transpose(self.parameters["W1"],0 , 1))
        linearop1 = linearop1 + self.parameters["b1"]
        nonlinearop1 = torch.zeros(self.parameters["b1"].size(dim=0))
        if (self.f_function == 'relu'):
            nonlinearop1 = torch.nn.functional.relu(linearop1)
        elif (self.f_function == 'sigmoid'):
            nonlinearop1 = torch.nn.functional.sigmoid(linearop1)
        else:
            nonlinearop1 = linearop1
        linearop2 = torch.matmul(nonlinearop1, torch.transpose(self.parameters["W2"],0 , 1))
        linearop2 = linearop2 + self.parameters["b2"]
        nonlinearop2 = torch.zeros(self.parameters["b2"].size(dim=0))
        if (self.g_function == 'relu'):
            nonlinearop2 = torch.nn.functional.relu(linearop2)
        elif (self.g_function == 'sigmoid'):
            nonlinearop2 = torch.nn.functional.sigmoid(linearop2)
        else:
            nonlinearop2 = linearop2
        
        return nonlinearop2

    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        pass

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    diff = y - y_hat
    N = y.size(dim=0) # batch size
    M = y.size(dim=1) # Feature size
    grad = -2 * diff / (N * M)
    return (torch.sum(torch.square(diff)) / (N * M), grad)

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    N = y.size(dim=0) # batch size
    M = y.size(dim=1) # Feature size
    loss = -torch.sum(torch.multiply(y, torch.log(y_hat)) + torch.multiply(1-y, torch.log(1-y_hat))) / (N*M)
    grad = (-y/y_hat + (1-y)/(1-y_hat)) / (N*M)
    return loss, grad

print("Torch version : ", torch.__version__)









