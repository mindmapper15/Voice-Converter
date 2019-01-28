# Voice-Converter

This program is based on by https://github.com/andabi/deep-voice-conversion

Original Author :

[Dabi Ahn](https://github.com/andabi)(andabi412@gmail.com)  

[Kyubyong Park](https://github.com/Kyubyong) (kbpark.linguist@gmail.com)

# 실행 환경

Ubuntu 16.04 LTS 64bit
Python 3.5 64bit

# 필요 라이브러리

- tensorflow == 1.8
- numpy == 1.15.0
- librosa == 0.5.1
- joblib == 0.11
- tensorpack == 0.8.6
- pyyaml
- soundfile
- pydub
- tqdm
- pyworld
- lws
- matplotlib
- sounddevice
- colorama
- keyboard

## 주의!
- 파이썬 버전을 반드시 확인하세요!
  파이썬 버전 확인은 python --version으로 확인 하실 수 있습니다.
  해당 명령어를 실행했을 때 파이썬 2.x.x로 버전이 확인된다면
  프로그램을 실행하실 때 반드시 python3 로 실행하셔야 합니다!

- 해당 프로그램은 실행 시, 루트 권한을 요구합니다.
  터미널에서 sudo su 명령어로 루트 권한을 준 뒤 해당 코드를 실행하시거나
  해당 코드를 실행 시 앞에 sudo 를 추가하여 루트 권한으로 실행하시기 바랍니다.

# 실행 방법

터미널을 열어서 change_model.py 코드를 실행합니다.


    python change_model.py -gpu 0 case1


또는


    python3 change_model.py -gpu 0 case1



각 명령어들의 의미는 다음과 같습니다.

- -gpu (GPU ID) : 텐서플로우를 GPU 환경에서 실행하실 경우, ID 번호들을 입력해 텐서플로우를 가동할 GPU를 선택할 수 있습니다.
                  텐서플로우는 기본적으로 사용자의 컴퓨터에 있는 모든 GPU 상에서 텐서플로우를 가동합니다.
                  자신의 컴퓨터에 설치된 GPU의 개수와 각각의 ID는 터미널에서 nvidia-smi 라는 명령어를 입력해 확인 가능합니다.

- case1 : 음성 인식기 모델의 이름입니다. 그대로 쓰시면 됩니다.



프로그램을 작동했으면 터미널에 뜨는 메세지에 따라 명령어를 입력하신 후
지시대로 실행하시면 됩니다.

(2018-11-17 : 모델 변경 기능을 추가했습니다. 현재는 4가지 모델을 사용 가능합니다)
