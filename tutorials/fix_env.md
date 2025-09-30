This guide is for fixing issue with `requests` and `urllib3`

- Run these in terminal:
```
python - <<'PYCODE'
import sys, site, glob, shutil
for path in site.getsitepackages() + [site.getusersitepackages()]:
    for pkg in glob.glob(f"{path}/urllib3*"):
        print("Removing:", pkg)
        shutil.rmtree(pkg, ignore_errors=True)
PYCODE
```

```
python - <<'PYCODE'
import sys, site, glob, shutil
for path in site.getsitepackages() + [site.getusersitepackages()]:
    for pkg in glob.glob(f"{path}/requests*"):
        print("Removing:", pkg)
        shutil.rmtree(pkg, ignore_errors=True)
PYCODE
```
- Update pip & wheel
`pip install --upgrade pip setuptools wheel`
- Reinstall 
`pip install urllib3 requests==2.31.0 --force-reinstall --no-cache-dir`