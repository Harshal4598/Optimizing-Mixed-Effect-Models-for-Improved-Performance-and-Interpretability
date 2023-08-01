# LOG

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
