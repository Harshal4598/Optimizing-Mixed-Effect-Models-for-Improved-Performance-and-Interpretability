# LOG

## 2024-02-26
## Weeks Left: 4

 - Agenda
    - Feedback on Mock-Defence
    - Check out Overleaf Format

 - Next steps
    - [ ] Documentation

## 2024-02-19
## Weeks Left: 5

 - Agenda
    - Discussion: Pre-Defence
    - Table of Contents

 - Next steps
    - [ ] Focus on writing


## 2024-02-16
## Weeks Left: 6

 - Agenda
    - Method: Heuristic Approach - Performance-based Pair Selection 
    - Presentation
    - Discussion: Improving Interpretability or prove trade-offs?


 - Next steps
    - [ ] Focus on writing

## 2024-02-05
## Weeks Left: 7

 - Agenda
    - Quick update with Brute Force Algorithm
    - Discuss: Algorithm of Informed Brute Force with shapley values 
               and goodness of variance fit

 - Next steps
    - [ ] Working Informed Brute Force
    - [ ] Interpretability


## 2024-01-29
## Weeks Left: 8

 - Agenda
    - Brute Force Algorithm
    - Simulation Results
    - Report and Table of Contents

 - Next steps
    - [ ] Method 4 : Brute Force
    - [ ] Informed Brute Force
    - [ ] Interpretability


## 2024-01-22
## Weeks Left: 9

 - Agenda
    - Method 1: Using Jenks Natural Breaks with PCA
    - Method 2: Using Clustering Algorithm with PCA
    - Method 3: Using Shapley values with PCA on Jenks / Clustering
    - Discuss : Brute Force & Interpretability


 - Next steps
    - [ ] overleaf
        - [ ] table of contents
        - [ ] detailed structure experiments/results
    - [ ] Mid-term report
    - [ ] Method 4 : Brute Force
        - [ ] heuristic/informed brute force
       - [ ] Sensible merging
    - [ ] agenda


## 2024-01-15
## Weeks Left: 10

 - Agenda
    - Future Steps

 - Next steps
    - [ ] Consider multi dimentional data with multiple slopes
    - [ ] Methods
            - Jenks with PCA
            - Clustering with PCA
            - Shapley with PCA
            - Brute Force 
            - Informed Brute Force with Shapley value
    - [ ] Mid-term Report

## 2024-01-08
## Weeks Left: 11

 - Agenda
    - Performance Comparison of Effective Groups and Visible Groups
        - Performance(Effective_Groups) > Performance(Visible_Groups)
        - Models may break as the number of visible groups increase.
    - Jenks Natural Breaks: Train-Test data performance diagnosis on knee plot
    - Clustering Methods to reduce Group
    - Discuss Interpretability

- Next steps
    - [ ] think about what we did, what we can do, and what we can do next
    - [ ] come up with a plan of what you think are the best next steps (be creative!)
    - [ ] start slides for midterm presentation
    - [ ] Work on Interpretability

## 2023-12-20
## Weeks Left: 14

 - Agenda
    - Method to reduce Group: Jenks Natural Breaks
    - Discussion/Confusion: Our Initial Hypothesis does not satisfy.
        (The model performs better when there are less groups.)

- Next steps
    - [ ] Improve the current Method to reduce Group.
    - clustering
        - [ ] intercept / slope plot
        - [ ] check clustering methods
        - [ ] train test test data split ... for knee plot
    - MixedLM
        - [ ] clean notebook
            - parameters at the top to play with
            - visualization of the data
            - plot only effective groups
            - plot visual groups
            - permutation of group labels   

## 2023-12-07

- next steps
    - [ ] fix the noise
    - [ ] use the right mixedlm formula
    - [ ] figure why predictions of mixedlm don't work on effective groups
    - [ ] produce those visible group figures with just two and three effective groups
    - [ ] brute force 

## 2023-11-16

- Agenda
    - Np.linspace on ARMED
    - Reason for ups and downs in MSE
    - Interpretability


- TODO

    - [ ] Prepare presentation
    - [ ] Work further on Interpretability


## 2023-11-06

- Agenda
    - Discussion about results and any improvement
    - Data Generation Algorithm Slides
    - Interpretability now or later?

- TODO

    - [ ] Work on slides for next group meeting
    - [ ] linspace (for something other than MERF)
    - [ ] interpretability
    
    - [ ] Start with Effective groups finding approach

## 2023-10-24

- Agenda
    - Abstract
    - Interpreting the results with better better slope generation
    - Discussion about effective Group Finding Approach


- TODO
    - [ ] Work on concrete results
    - [ ] Start with Effective groups finding approach



## 2023-10-16

- Agenda
    - Results - “slopes” and “both” mode with 10 Samplings
    - Interpreting the results
    - Abstract
    - Discussion about effective Group Finding Approach

- TODO
    - [ ] Perform basic Group Finding Approach - BFS
    - [ ] Look for other approaches


## 2023-10-11

- Agenda
    - Random Slope Results discussion
    - Data generation - Random Intercept & slopes

- TODO
    - [ ] update visualization with sampling (one of 2 versions)
    - [ ] Simulate for Random Intercept and Slopes
    - [ ] Start with effective group finding approach
    - [ ] Abstract


## 2023-09-26

- Agenda
    - Finalizing Data Generation Algorithm
    - Random slopes with ARMED and others
    - Discussion: At Least need one model for Random Slopes    
    - Abstract

- TODO
    - [ ] Update Abstarct
    - [ ] Find ateast one working model for random slopes

## 2023-09-11

- notes
    - even if things still work with many visible groups 
        we have an interpretation issue
- TODO
    - [ ] update data generation
    - [ ] same figure with updated data generation
    - [ ] random slopes (ARMED)
    - [ ] random slopes + intercepts
    - [ ] abstract
    - [ ] try with interaction terms

## 2023-09-04

- agenda
    - Confusion with Differentiating real groups
    - Some thoughts with related work papers
- TODO
    - [ ] Intercepts performance plot (real Groups <-> effective groups)
    - [ ] abstract 

## 2023-08-29

- agenda
    - New ways to generate mixed effcts - Intercepts, Slopes and Mixed
- TODO
    - [ ] Perform it for intercepts
    - [ ] abstract

## 2023-08-28 - Group Meeting

- agenda
    - Group Search Approaches
    - What about EMM - SD now?
    - If this suffice then, Non-linaer or Classification?
- TODO
    - [ ] abstract (will send an email before next meeting)
    - [ ] Solid proof - Slope model does not work or too complicated
    - [ ] 
    - [ ] make comparison table (interpretable)

## 2023-08-23

- agenda
- TODO
    - [ ] abstract
    - [ ] make comparison table (interpretable)
    - [ ] do it for 20 groups (if it does not run too long)
    - [ ] 
    - [ ] agenda + next steps


## 2023-08-07 (8/4)

- agenda
    - LMMNN - Random intercepts with multiple categorical features
    - Discussion about the objective

- TODO
    - [ ] Random intercepts with multiple groups until it breaks for LMMNN and MixedLM
    - [ ] Get random slopes running

## 2023-08-01 (7/4)

- agenda
    - LMMNN - Random Intercepts
    - Complications with LMMNN Random Slopes
    - Neural Network used in LMMNN
    - Group Meeting
- TODO
    - [ ] Random intercepts with multiple groups untill it breaks for LMMNN and MixedLM
    - [ ] Get random slopes running

## 2023-07-11 (6/4)

- agenda
    - lmmnn performance issues
    - synthetic data with regression and mixedlm
    - focus > regression or classification?
- TODO
    - [ ] lmmnn linear, no hidden layers
    - [ ] mixedlm: add binary columns (cartesian) until ite breaks down
    - [ ] agenda + summary of work + next steps (two to three items)

## 2023-06-27 (Pre 5/4)

- agenda
    - show some results
        - synthetics
    - talk about lmmnn
- TODO
    - [ ] agenda + next steps
    - [ ] slides (five minutes)
    - [ ] ARMED

## 2023-06-20 (Pre 4/4)

- agenda
    - go through TODO
    - subgroups
    - datasets
- PLAN
    - [ ] lmmnn/mixedlm on group combinations: report differences
        - [ ] try all combination
        - [ ] synthetic dataset where subset of groups is better
    - [ ] lmmnn run more examples
    - [ ] try ARMED again
        - [ ] only classification?


## 2023-06-13 (Pre 3/4)

- agenda
    - Subgroup Discovery
    - what about mixedlm (lmmnn)
    - ARMED Solution
- notes
    - subgroup discovery
    - lmmnn: still coding
- TODO
    - [ ] try to run lmmn
    - [ ] try to find data sets for fixed effects modeling but with more than one variable for groups
    - [ ] for practice: try to implement
        - [Understanding Where Your Classifier Does (Not) Work -- The SCaPE Model Class for EMM](https://wwwis.win.tue.nl/~wouter/Publ/C11-SCaPE.pdf)
        - to do this have a look at [`binary_target.py`](https://github.com/flemmerich/pysubgroup/blob/master/pysubgroup/binary_target.py) and implement your own target
        - I added a very simple implementation of something similar (not nicely programmed) to this repo (this may already implement what we want, but please check it)
        - test it with some classification task
        - the idea is later to create our own target sepcific to slecting groups of variables (not instances) for mixed models using subgroup analysis

## 2023-06-07 (Pre 2/4)

- agenda
    - Use of Design Matrix in Classification Problem
    - Subgroup Discovery, the results and more
    - BinomialBayesMixedGLM, reading reults and ...
    - Any Advice and Feedback
- notes
    - [x] pysubgroup
- TODO
    - [ ] play around with a parameter of standard quality function
    - [ ] make small notebooks ... with simple example

## 2023-05-31 (Pre 1/4)

- agenda
    - ARMED mixed effects models
    - other model
- TODO
    - [ ] ARMED
    - [ ] mixedlm
    - [ ] pysubgroup
    - [ ] agenda

## 2023-05-19

- TODO
    - try run at least 3 mixed effects with different datasets (including ARMED)
