# Import 문제 해결 가이드

BSMamba2 프로젝트에서 `ModuleNotFoundError: No module named 'data'` 또는 유사한 오류가 발생하는 경우, 다음 해결 방법들을 시도해보세요.

## 해결 방법 1: run.py 래퍼 스크립트 사용 (권장)

프로젝트 루트에서 `run.py`를 사용하여 예제 스크립트를 실행하세요:

```bash
# 학습 예제 실행
python run.py examples/train_example.py

# 빠른 시작 예제 실행
python run.py examples/quick_start.py --model checkpoints/model.pt --input audio.wav --output vocals.wav

# 평가 예제 실행
python run.py examples/evaluate.py --model checkpoints/model.pt --musdb-root /path/to/musdb18hq
```

## 해결 방법 2: PYTHONPATH 환경 변수 설정

### Linux/Mac:
```bash
export PYTHONPATH=/path/to/Mamba2MeetsSilence:$PYTHONPATH
python examples/train_example.py
```

### Windows PowerShell:
```powershell
$env:PYTHONPATH="M:\Diff\MMS;$env:PYTHONPATH"
python examples/train_example.py
```

### Windows CMD:
```cmd
set PYTHONPATH=M:\Diff\MMS;%PYTHONPATH%
python examples/train_example.py
```

## 해결 방법 3: 패키지 설치 (개발 모드)

프로젝트를 editable 모드로 설치하면 import 문제가 완전히 해결됩니다:

```bash
cd /path/to/Mamba2MeetsSilence
pip install -e .
```

이후에는 어디서든 스크립트를 실행할 수 있습니다:

```bash
python examples/train_example.py
```

**참고**: 권한 오류가 발생하는 경우:
- Linux/Mac: `sudo pip install -e .` 또는 가상 환경 사용
- Windows: 관리자 권한으로 실행하거나 가상 환경 사용

## 해결 방법 4: 직접 프로젝트 디렉토리에서 실행

프로젝트 루트 디렉토리에서 Python 모듈로 실행:

```bash
cd /path/to/Mamba2MeetsSilence
python -m examples.train_example
```

## Import 테스트

모든 모듈이 정상적으로 import되는지 확인:

```bash
python test_imports.py
```

모든 import가 성공하면 ✓ 표시가 나타납니다.

## 예제 스크립트 구조

모든 예제 스크립트(`examples/*.py`)는 다음과 같이 프로젝트 루트를 Python 경로에 추가합니다:

```python
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
```

이 코드는 스크립트가 어디서 실행되든 프로젝트 모듈을 찾을 수 있도록 합니다.

## 문제가 계속되는 경우

1. Python 버전 확인 (>= 3.8 권장):
   ```bash
   python --version
   ```

2. 현재 작업 디렉토리 확인:
   ```bash
   pwd  # Linux/Mac
   cd   # Windows
   ```

3. 프로젝트 구조 확인:
   ```bash
   ls -la  # Linux/Mac
   dir     # Windows
   ```

4. test_imports.py 실행하여 어떤 모듈에서 문제가 발생하는지 확인

5. GitHub Issues에 문제 보고 (에러 메시지와 환경 정보 포함)

## 가상 환경 사용 (권장)

프로젝트를 격리된 환경에서 실행하려면:

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 프로젝트 설치 (editable 모드)
pip install -e .

# 이제 예제 실행
python examples/train_example.py
```
