pdoc nlp_uncertainty_zoo/*/*.py --docformat numpy --logo "https://raw.githubusercontent.com/Kaleidophon/nlp-uncertainty-zoo/main/img/logo.png" --o docs
grep -o '<li><a href=".*">.*</a></li>' docs/index.html > modules.txt
pdoc nlp_uncertainty_zoo/__init__.py --logo "https://raw.githubusercontent.com/Kaleidophon/nlp-uncertainty-zoo/main/img/logo.png" --o docs
