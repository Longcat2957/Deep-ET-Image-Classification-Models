import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional


def autopad(k: Union[int, List[int]], p: Union[int, List[int]] = None, d: int = 1) -> Union[int, List[int]]:
    """
    주어진 커널 크기 k, 패딩 크기 p, 스트라이드 d를 사용하여 'same' 모양의 출력을 얻기 위해 필요한 패딩 크기 p를 반환합니다.
    
    Args:
        k: 합성곱 커널의 크기입니다. 정수인 경우 해당 크기의 정사각형 커널을 가정합니다.
           리스트인 경우 해당 모양의 커널을 가정합니다.
        p: 추가할 패딩 양입니다. 정수인 경우 모든 면에 대해 동일한 패딩이 추가됩니다.
           리스트인 경우 각 면에 해당 크기의 패딩이 추가됩니다. 기본값은 None입니다.
           None인 경우 'same' 모양의 출력을 얻기 위한 패딩을 계산합니다.
        d: 합성곱 연산의 스트라이드입니다.
    
    Returns:
        패딩 크기를 나타내는 정수 또는 정수 리스트입니다.
    """
    
    # 만약 스트라이드가 1보다 크면, 유효한 커널 크기를 계산합니다.
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else \
            [d * (x - 1) + 1 for x in k]
    
    # 만약 패딩이 지정되지 않은 경우, 'same' 모양의 출력을 얻기 위해 패딩을 계산합니다.
    if p is None:
        p = k // 2 if isinstance(k, int) else \
            [x // 2 for x in k]
            
    return p

class Conv(nn.Module):
    '''
    채널입력, 채널출력, 필터크기, 스트라이드, 패딩, 그룹, dilation, 활성화함수 를 인자로 받아 표준 합성곱 레이어를 생성합니다.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int, optional): Kernel size. Default is 1.
        s (int, optional): Stride. Default is 1.
        p (int, optional): Padding. Default is None.
        g (int, optional): Number of groups. Default is 1.
        d (int, optional): Dilation. Default is 1.
        act (bool or nn.Module, optional): Whether to use activation function. Default is True.

    Attributes:
        conv (nn.Conv2d): 2D convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass with batch normalization and activation.
        forward_fuse(x: torch.Tensor) -> torch.Tensor: Forward pass without batch normalization.
    '''

    default_act = nn.ReLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass with batch normalization and activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        '''
        return self.act(self.conv(x))