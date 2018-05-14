mkdir repository
cd repository
git init
touch 1
git add 1
git commit -m 'Add file 1'
touch 2
git add 2
git commit -m 'Add file 2'
touch 3
git add 3
git commit -m 'Add file 3'
touch 4
git add 4
git commit -m 'Add file 4'
touch 5
git add 5
git commit -m 'Add file 5'
##
git checkout -b feature HEAD@{4}
touch 6
git add 6
git commit -m 'Add file 6'
touch 7
git add 7
git commit -m 'Add file 7'
touch 8
git add 8
git commit -m 'Add file 8'
##
git checkout master
git rebase HEAD~2 --onto feature
git checkout -b debug HEAD@{11}
touch 9
git add 9
git commit -m 'Add file 9'

git --no-pager reflog > reflog.txt
