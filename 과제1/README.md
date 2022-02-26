### 작업환경
작업환경
OS : Windows 10
IDE : Visual Studio 2019
GPU : GeForce GTX 1050 Ti


### 프로젝트 사용방법
기본적으로 그냥 컴파일을 하면 되지만, Config_CBO.h에서 #define된 KERNEL_SELECTION의
숫자를 바꿔서, openCL 계산방법을 바꾸고, GAUSSIAN_FILTER_SIZE를  바꿔서
필터 사이즈를 바꿀 수 있다. 여기에 LOCAL_WORK_SIZE_0, LOCAL_WORK_SIZE_1을
바꿔서 work group size를 바꿀 수 있다. (당연히 2의 지수단위로 바꿔야 함. 현재 
maximum size는 256이므로 둘이 곱해서 256을 넘으면 안된다.)

