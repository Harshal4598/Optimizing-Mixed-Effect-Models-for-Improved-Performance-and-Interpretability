# LOG

## 2023-11-06

- Agenda
    - Discussion about results and any improvement
    - Data Generation Algorithm Slides
    - Interpretability now or later?

- TODO

    - [ ] Work on slides for next group meeting
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
