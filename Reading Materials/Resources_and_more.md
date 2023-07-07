# This is a small list of needed resources for the project

## Numba
First, [numba](https://numba.readthedocs.io/en/stable/user/5minguide.html). It allows to compile python code, and makes it faster by allowing it to be parallelized on the cpu (and CUDA gpu, but that's for later.). 

The main tools we will need initially for CPU parallelization are located in [here](https://numba.readthedocs.io/en/stable/user/parallel.html). Essentially, we will just parallelize the loop used to update the state, using the decorator "@njit(parallel=true)", and using prange instead of range for the loops.

The code that is written will basically be python code, but with a decent amount of restrictions.

Workflow :
1. Make the code work without any parallelization (without the njit decorator).
2. Add the parallelization, fix the bugs introduced by it.
3. Eventually might add also GPU support, but that's for later.

## Git
It will be a bit more practical to use git to share the code more effectively. We won't need any advanced use of it. All you need to know essentially :

- To clone the repository, git clone \<link\>. 
- Once in a repository, you can show all existing branches (local and remote) with `git branch -a`. We will usually work on a single branch to make things easier. For the project this is the branch `nophoton`.
- Once inside a branch, do a `git pull` to get the latest update from the 'cloud' repository. If there are problems here, we can talk about them, usually they are resolved easily.
- To move to another branch, simply type `git checkout <branchname>`.
- You can use `git status` to see which branch you are on, and what are the files that have been modified/deleted. Files appearing in red are 'unstaged', meaning not added to the next commit yet.
- To 'stage' files for a commit, you can do `git add <filename>`. Usually we just add all the files, so we just do `git add -A`.
- If you type `git status` again, the modifications should appear in green.
- Once the files are stages, you can do `git commit -m "<commit message>"`. This will add the commit (locally), with the message.
- To "push" the changes to the remote(cloud) repository, so that other collaborators can 'pull' the changes, simply type `git push -u`. This can sometimes fail with an error, usually because the remote branch does not exist, simply follow the instruction of the error.
- And that's it ! If we don't deal with several branches and all that, things should remain smooth.

## VSCode
- In VSCode, you should install the `python` extention if not done yet, and that's pretty much it.
- Using the debugger (by launching the main.py pressing F5, or pressing the play button in the debugger side window (open manually or CMD+shift+D)). You can put debugger 'stops' by clicking to the left of the line numbers, which will add a red dot. When stopped in debugging, in the debuggin window you can see the value of all the variables which are defined. 
- Of course, debugging by adding print statements also works fine.
- When running the code in parallel mode with Numba, the error messages might be more cryptic. Always 'turn off' parallel computing if you suspect there is a bug with the code that does not depend on the parallelization.

## Pygame
- We use pygame just as a visualization tool, which allows for interactivity. All the pygame code is contained in the main.py (and Camera.py). The main instanciates a SMCA class, and uses it to step at each frame, and display the worldmap. Documentation is [here](https://www.pygame.org/docs/), although often it's easier to ask ChatGPT.