mkdir -p ~/.streamlit/
echo "\
[server]\n\git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml