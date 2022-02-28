### 작업 환경
OS : Windows 10
IDE : Visual Studio 2019
GPU : GeForce GTX 1050 Ti

### 프로젝트 사용 방법

##### 1. 파일 선택
Config_CBO.h의 INPUT_IMAGE의 값을 0 ~ 9중 하나로 설정합니다.

##### 2. 필터 크기
Config_CBO.h의 GAUSSIAN_FILTER_SIZE를 5, 7, 9, 11 중 하나로 설정합니다. 
GPU연산 수행시에는 KERNEL_SELECTION의 값 또한 선택한 (필터크기+10), 또는 (필터크기+20)로 설정합니다.

##### 3. CPU mode vs. GPU mode
GPU mode의 경우, predefined한 KERNEL_SELECTION의 값인 0, 1, 11, 15, 17, 19, 111, 2, 25, 27, 28, 211 중 하나로 설정합니다.
CPU mode의 경우, 위에서 나온 값이 아닌 다른 값으로 설정합니다.
