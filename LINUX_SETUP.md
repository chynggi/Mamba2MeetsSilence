# Linux/Unix 환경에서 BSMamba2 실행하기

Linux 또는 Unix 계열 시스템에서 BSMamba2를 실행하는 방법입니다.

## 빠른 시작

### 1단계: PYTHONPATH 설정 및 테스트

```bash
cd /workspace/Mamba2MeetsSilence

# PYTHONPATH 설정
export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH

# Import 테스트
python3 test_imports.py
```

### 2단계: 예제 실행

```bash
# 학습 예제
python3 examples/train_example.py

# 또는 run.py 래퍼 사용
python3 run.py examples/train_example.py
```

## 영구적인 PYTHONPATH 설정

매번 export를 실행하지 않으려면, 쉘 설정 파일에 추가하세요:

### Bash (.bashrc 또는 .bash_profile)

```bash
echo 'export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### Zsh (.zshrc)

```bash
echo 'export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH' >> ~/.zshrc
source ~/.zshrc
```

## 가상 환경 사용 (권장)

```bash
# 프로젝트 디렉토리로 이동
cd /workspace/Mamba2MeetsSilence

# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# Editable 모드로 프로젝트 설치
pip install -e .

# 이제 어디서든 실행 가능
python examples/train_example.py
```

## 빠른 설정 스크립트 사용

```bash
# 스크립트에 실행 권한 부여
chmod +x setup_and_test.sh

# 실행
./setup_and_test.sh
```

## Docker 환경에서 실행

Docker 컨테이너 내에서 실행하는 경우:

```bash
# 컨테이너에서 PYTHONPATH 설정
export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH

# 또는 Docker 실행 시 환경 변수 추가
docker run -e PYTHONPATH=/workspace/Mamba2MeetsSilence your-image
```

## 문제 해결

### "ModuleNotFoundError: No module named 'data'" 오류

```bash
# 1. 현재 디렉토리 확인
pwd
# 출력: /workspace/Mamba2MeetsSilence

# 2. PYTHONPATH 확인
echo $PYTHONPATH
# /workspace/Mamba2MeetsSilence가 포함되어야 함

# 3. 프로젝트 구조 확인
ls -la
# data/, models/, training/, utils/ 디렉토리가 보여야 함

# 4. Import 테스트
python3 test_imports.py
```

### Python3가 아닌 python 명령 사용

일부 시스템에서는 `python` 명령을 사용해야 할 수 있습니다:

```bash
# python 명령 확인
which python
which python3

# alias 설정 (선택사항)
alias python=python3
```

## 실행 옵션 요약

실행 방법을 선호도 순으로 나열:

1. **가상 환경 + pip install -e .** (가장 권장)
   ```bash
   python examples/train_example.py
   ```

2. **PYTHONPATH + 직접 실행**
   ```bash
   export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH
   python3 examples/train_example.py
   ```

3. **run.py 래퍼 사용**
   ```bash
   python3 run.py examples/train_example.py
   ```

4. **Python 모듈로 실행** (실험적)
   ```bash
   cd /workspace/Mamba2MeetsSilence
   python3 -m examples.train_example
   ```

## 자동화 스크립트 예제

학습을 자동으로 실행하는 스크립트:

```bash
#!/bin/bash
# train.sh

cd /workspace/Mamba2MeetsSilence
export PYTHONPATH=/workspace/Mamba2MeetsSilence:$PYTHONPATH
python3 examples/train_example.py "$@"
```

사용법:
```bash
chmod +x train.sh
./train.sh
```

## 추가 지원

문제가 계속되면:
1. `test_imports.py` 실행 결과 확인
2. `TROUBLESHOOTING.md` 문서 참조
3. GitHub Issues에 보고
