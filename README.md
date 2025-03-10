# Expected Possession Value Prevented
This contains the code used in the creation of a new defensive metric called Expected Possession Value prevented which was presented at the 2025 American Soccer Insights Summit. The dataset consists of Skillcorner tracking data from the 2024 National Women's Soccer League season and the 2023-2024 English Women's Super League as well as Wyscout event data synced to the tracking data for those seasons.

The website for the conference is here: https://americansoccerinsights.com/

A recording of my presentation is available here: https://youtu.be/ts0veG2_doQ?si=1aSUpptvF4b0L1jN 

A summary of the conference including a short discussion of this work written for American Soccer Analysis by Akshay Easwaran can be found here: https://www.americansocceranalysis.com/home/2025/3/3/a-day-at-the-american-soccer-insights-summit-the-latest-and-greatest-in-soccer-analytics

Likewise a summary written by Phil West for Verde All Day can be found here: https://verdeallday.com/american-soccer-insights-summit-report-rice-university/ 

You can find Skillcorner's open data to see data similar to what I was working with looks like here: https://github.com/SkillCorner/opendata
# Introduction:
Measuring defensive skill is a known hard problem in soccer analytics. There are generally two commonly thought of aspects of defensive skill:

1.) The ability to win the ball via a tackle or interception and 

2.) The ability to prevent dangerous passes and runs via positioning.

Winning the ball back is extremely valuable, but is also high risk. A missed tackle frequently results in the opposing team breaking a line of pressure. Measuring the effect of a defenders off-ball positioning on mitigating risk is extremely difficult, even with access to tracking data. This project originates in the idea that we may be missing something in between the above two ideas.
As coaches, players, and fans we recognize that not every defensive engagement with a ball-carrier will result in an attempted tackle or interception. We also recognize that this is frequently not a failure of the defender. In fact, there are many scenarios where it is likely a better option to simply show the attacker to a less dangerous area of the pitch or force a less dangerous pass. In this project, I attempt to quantify how good a player is at preventing dangerous play without necessarily winning the ball back.
# The model:
Using the data mentioned previously, I defined a 1v1 situation to be one where at the time of the event the closest defender to the ball-carrier was within 4 meters and the second closest defender was futher that 8 meters. This definition is purely heuristic and can likely be improved. A semi-supervised approach where a subset of the data is labeled and then a model iteratively labels the rest of the data would likely be effective here. A rules based approach which used the tracking data to take into account what the other defenders are doing would also likely be an improvement.
I trained an XGBoost regression model to predict the Expected Possession Value (EPV) added by the next event in a sequence of two successful passes. For now the model only takes as input the x,y location of the event and the two prior passes as well as the EPV added by the two prior passes. This is in line with the models built in the papers: "What Happened Next? Using Deep Learning to Value Defensive Actions in Football Event-Data" (https://arxiv.org/abs/2106.01786) and "Improving The Evaluation of Defensive Player Values with Advanced Machine Learning Techniques" (https://aisel.aisnet.org/isd2014/proceedings2024/datascience/24/). A first and most effective improvement to the model would likely come from adding more contextual information to the input of this model. These could be derived from the tracking data.
The metric, Expected Possession Value prevented (EPVp) is calculated as the difference between predicted EPV added and actual EPV added.
$$ EPVp = (predicted EPV added) - (EPV added)$$
# The code:
The files Skillcorner_IO_pub, ext_EPV_model_training_pub, and Get_1v1s_pub contain the code used to train the model and generate the data used in the presentation given at ASI. A longer blog post is in the works to explain this work and explore it further. The slides of the presentation are available on this page.
