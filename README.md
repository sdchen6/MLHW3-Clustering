# HW 3: Clustering for CS 349 @ NU
**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's NetID 
here, we will check. The NetID should be placed into that file by itself as the only line, 
with lower case letters and no carriage returns. 

The line that is marked NETID_GOES_HERE should be **REPLACED** with YOUR NetID.

**IF YOU DON'T PUT YOUR NETID INTO THAT FILE PROPERLY YOU WILL NOT RECIEVE CREDIT FOR YOUR WORK!**

In this assignment, you will:
- Implement k-Means
- Implement Gaussian Mixture Models (GMMs)
- Apply the two approaches to a dataset to discover patterns in an unsupervised way.
  
## Clone this repository

To clone this repository install GIT on your computer and copy the link of the repository (find above at "Clone or Download") and enter in the command line:

``git clone YOUR-LINK``

Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit and what goes into the write-up.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Environment setup

Make a conda environment for this assignment and then run:

``pip install -r requirements.txt``

## Running the test cases

The test cases can be run with:

``python -W ignore -m pytest -s .``

at the root directory of the assignment repository.  NOTE: this command is slightly changed
to suppress warnings that come from scikit-learn when we use adjusted_mutual_info.
