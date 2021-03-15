#rsync -avz --exclude-from='.gitignore' --exclude '.git/' . hansonlu@madmax.stanford.edu:remote/qagnn_generative
my_rsync () {
    git ls-files --exclude-standard -oi --directory > .git/ignores.tmp
    rsync -avzh --exclude ".git/" --exclude-from=".git/ignores.tmp" $@
}

my_rsync . hansonlu@madmax.stanford.edu:remote/FLAG_repo/

