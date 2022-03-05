### 작업환경

OS : Windows 10
IDE : Visual Studio 2019
GPU : GeForce GTX 1050 Ti

### 프로젝트 사용방법

(컴파일 및 실행은 안다고 가정을 합니다.)
#### 1. Simulation method 설정
Config_WE.h 에서 19행에 있는 SIMULATION_METHOD 값을 0, 1, 2중 하나로 설정합니다.
0 선택 시 GPU_EXPLICIT
1 선택 시 GPU_IMPLICIT_JACOBI
2 선택 시 CPU_IMPLICIT_JACOBI를 수행합니다.

#### 2. Simulation parameters
Config_WE.h의 21행에 있는 SIMULATION_PARAMETERS 값을 {0,1,2}중 하나로 설정하여 시뮬레이션 속도를 조절할 수 있습니다.

#### 3. Number of simulation steps
Config_WE.h의 23행에 있는 NUMBER_OF_SIMULATION_STEPS 값을 설정하여 시뮬레이션 수행 시간을 설정할 수 있습니다.

프로그램 실행 시 s키를 누르면 wave animation이 진행되는 것을 볼 수 있고, 토글 방식으로 한 번 누르면 정지 다시 누르면 
진행함.
z키를 누르면 확대가 가능하고 처음은 가장 작은 구도로 초기화 되고, 어느 정도 누르면 다시 초기의 상태로 돌아갑니다.
