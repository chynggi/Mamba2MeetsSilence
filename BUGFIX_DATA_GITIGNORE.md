# 🐛 Bug Fix: data 폴더 Git 추적 문제 해결

## 문제 상황

Linux 서버에서 다음과 같은 import 오류가 발생했습니다:
```
ModuleNotFoundError: No module named 'data'
```

Windows 로컬에서는 정상 작동하지만, Linux 서버(Git clone 후)에서만 실패했습니다.

## 원인

`.gitignore` 파일에 `data/`가 포함되어 있어서:
- `data/` 폴더 전체가 Git에서 무시됨
- 로컬에는 `data/` 폴더가 있지만, Git 저장소에는 올라가지 않음
- Linux 서버에서 clone하면 `data/` 폴더 자체가 없어서 import 실패

**문제가 된 `.gitignore` 라인:**
```gitignore
# Datasets
musdb18hq/
data/          # ← 이것이 소스 코드 폴더까지 제외시킴
```

## 해결 방법

### 1. `.gitignore` 수정 ✅

`data/` 라인을 제거하고 주석 추가:

```gitignore
# Datasets
musdb18hq/
# Note: data/ is our source code package, not ignored
# If you have dataset files, put them in a different directory like datasets/
```

### 2. `data/` 폴더를 Git에 추가

```bash
git add data/
git commit -m "Fix: Add data package to Git (was incorrectly ignored)"
git push
```

### 3. 데이터셋 파일 관리 지침

- ✅ **소스 코드**: `data/` 폴더 (Git에 포함)
- ❌ **데이터셋 파일**: 별도 디렉토리에 저장 (`musdb18hq/`, `datasets/` 등)

## 교훈

1. **`.gitignore`의 광범위한 패턴 주의**: `data/`처럼 일반적인 이름은 의도하지 않은 파일까지 제외할 수 있음
2. **명확한 네이밍**: 
   - 소스 코드 패키지: `bsmamba2_data/` 또는 `src/data/`
   - 데이터셋 디렉토리: `datasets/`, `musdb18hq/` 등
3. **Git 추적 확인**: `git status`로 중요한 파일이 tracked 되는지 확인

## 검증

### Linux 서버에서 확인:

```bash
# 1. 최신 코드 pull
git pull

# 2. data 폴더 확인
ls -la data/
# 출력: __init__.py, dataset.py, transforms.py, README.md

# 3. Import 테스트
python test_imports.py
# 모든 import가 성공해야 함

# 4. 학습 실행
python examples/train_example.py
```

## 추가 도구

문제 진단을 위해 만든 도구들:

- `test_imports.py` - 모든 모듈 import 테스트
- `debug_data_import.py` - data 모듈 import 상세 디버깅
- `setup_imports.py` - Import 경로 자동 설정 헬퍼

## 관련 파일

- `.gitignore` - 수정됨
- `data/README.md` - 새로 추가 (용도 명시)
- `TROUBLESHOOTING.md` - Import 문제 해결 가이드
- `LINUX_SETUP.md` - Linux 환경 설정 가이드

---

**수정 날짜**: 2025년 9월 30일  
**수정자**: GitHub Copilot  
**이슈**: data 폴더가 .gitignore로 인해 Git에서 누락됨
