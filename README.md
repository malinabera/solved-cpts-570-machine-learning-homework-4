Download Link: https://assignmentchef.com/product/solved-cpts-570-machine-learning-homework-4
<br>



<strong>CptS 570 Machine Learning</strong>

<strong>Homework #4</strong>







<h1>Analytical Part (3 Percent + 1 Percent extra credit)</h1>

<ol>

 <li><strong>(Finite-Horizon MDPs.) </strong>Our basic definition of an MDP in class defined the reward function <em>R</em>(<em>s</em>) to be a function of just the state, which we will call a <em>state reward function</em>. It is also common to define a reward function to be a function of the state and action, written as <em>R</em>(<em>s,a</em>), which we will call a <em>state-action reward function</em>. The meaning is that the agent gets a reward of <em>R</em>(<em>s,a</em>) when they take action <em>a </em>in state <em>s</em>. While this may seem to be a significant difference, it does not fundamentally extend our modeling power, nor does it fundamentally change the algorithms that we have developed.

  <ol>

   <li>Describe a real world problem where the corresponding MDP is more naturally modeledusing a state-action reward function compared to using a state reward function.</li>

   <li>Modify the Finite-horizon value iteration algorithm so that it works for state-action rewardfunctions. Do this by writing out the new update equation that is used in each iteration and explaining the modification from the equation given in class for state rewards.</li>

   <li>Any MDP with a state-action reward function can be transformed into an “equivalent”MDP with just a state reward function. Show how any MDP with a state-action reward function <em>R</em>(<em>s,a</em>) can be transformed into a different MDP with state reward function <em>R</em>(<em>s</em>), such that the optimal policies in the new MDP correspond exactly to the optimal policies in the original MDP. That is an optimal policy in the new MDP can be mapped to an optimal policy in the original MDP. <em>Hint: It will be necessary for the new MDP to introduce new “book keeping” states that are not in the original MDP.</em></li>

  </ol></li>

 <li><strong>(</strong><em>k</em><strong>-th Order MDPs.) </strong>A standard MDP is described by a set of states <em>S</em>, a set of actions <em>A</em>, a transition function <em>T</em>, and a reward function <em>R</em>. Where <em>T</em>(<em>s,a,s</em><sup>0</sup>) gives the probability of transitioning to <em>s</em><sup>0 </sup>after taking action <em>a </em>in state <em>s</em>, and <em>R</em>(<em>s</em>) gives the immediate reward of being in state <em>s</em>.</li>

</ol>

A <em>k</em>-order MDP is described in the same way with one exception. The transition function <em>T </em>depends on the current state <em>s </em>and also the previous <em>k</em>−1 states. That is, <em>T</em>(<em>s<sub>k</sub></em><sub>−1</sub><em>,</em>···<em>,s</em><sub>1</sub><em>,s,a,s</em><sup>0</sup>) = <em>Pr</em>(<em>s</em><sup>0</sup>|<em>a,s,s</em><sub>1</sub><em>,</em>···<em>,s<sub>k</sub></em><sub>−1</sub>) gives the probability of transitioning to state <em>s</em><sup>0 </sup>given that action <em>a </em>was taken in state <em>s </em>and the previous <em>k </em>− 1 states were (<em>s<sub>k</sub></em><sub>−1</sub><em>,</em>···<em>,s</em><sub>1</sub>).

Given a <em>k</em>-order MDP <em>M </em>= (<em>S,A,T,R</em>) describe how to construct a standard (First-order)

MDP <em>M</em><sup>0 </sup>= (<em>S</em><sup>0</sup><em>,A</em><sup>0</sup><em>,T</em><sup>0</sup><em>,R</em><sup>0</sup>) that is equivalent to M. Here equivalent means that a solution to <em>M</em><sup>0 </sup>can be easily converted into a solution to <em>M</em>. Be sure to describe <em>S</em><sup>0</sup>, <em>A</em><sup>0</sup>, <em>T</em><sup>0</sup>, and <em>R</em><sup>0</sup>. Give a brief justification for your construction.

<ol start="3">

 <li>Some MDP formulations use a reward function <em>R</em>(<em>s,a</em>) that depends on the action taken in a</li>

</ol>

state or a reward function <em>R</em>(<em>s,a,s</em><sup>0</sup>) that also depends on the result state <em>s</em><sup>0 </sup>(we get reward <em>R</em>(<em>s,a,s</em><sup>0</sup>) when we take action <em>a </em>in state <em>s </em>and then transition to <em>s</em><sup>0</sup>). Write the Bellman optimality equation with discount factor <em>β </em>for each of these two formulations.

<ol start="4">

 <li>Consider a trivially simple MDP with two states <em>S </em>= {<em>s</em><sub>0</sub><em>,s</em><sub>1</sub>} and a single action <em>A </em>= {<em>a</em>}. The reward function is <em>R</em>(<em>s</em><sub>0</sub>) = 0 and <em>R</em>(<em>s</em><sub>1</sub>) = 1. The transition function is <em>T</em>(<em>s</em><sub>0</sub><em>,a,s</em><sub>1</sub>) = 1 and <em>T</em>(<em>s</em><sub>1</sub><em>,a,s</em><sub>1</sub>) = 1. Note that there is only a single policy <em>π </em>for this MDP that takes action <em>a </em>in both states.</li>

 <li>Using a discount factor <em>β </em>= 1 (i.e. no discounting), write out the linear equations for evaluating the policy and attempt to solve the linear system. What happens and why?</li>

 <li>Repeat the previous question using a discount factor of <em>β </em>= 0.9.</li>

 <li>Please read the following paper and briefly summarize (at most one page) the key ideas as you understood:</li>

</ol>

Thomas G. Dietterich (2000). Ensemble Methods in Machine Learning. J. Kittler and F. Roli (Ed.) First International Workshop on Multiple Classifier Systems, Lecture Notes in Computer Science (pp. 1-15). New York: Springer Verlag.

<a href="http://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf">http://web.engr.oregonstate.edu/</a><a href="http://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf">~</a><a href="http://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf">tgd/publications/mcs-ensembles.pdf</a>

<ol start="6">

 <li>Please read the following paper and write a brief summary (at most one page) of the main points.</li>

</ol>

Matthew Zook, Solon Barocas, danah boyd, Kate Crawford, Emily Keller, Seeta Pea Gangadharan, Alyssa Goodman, Rachelle Hollander, Barbara Knig, Jacob Metcalf, Arvind Narayanan,

Alondra Nelson, Frank Pasquale: Ten simple rules for responsible big data research. PLoS Computational Biology 13(3) (2017) <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/journal.pcbi_.1005399.pdf">https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/journal.pc</a>bi_ <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/journal.pcbi_.1005399.pdf">.1005399.pdf</a>

<ol start="7">

 <li>Please read the following paper and write a brief summary (at most one page) of the main points.</li>

 <li>Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-Franois Crespo, Dan Dennison: Hidden Technical Debt in Machine Learning Systems. NIPS 2015: 2503-2511</li>

 <li>Please read the following paper and write a brief summary (at most one page) of the main points.</li>

</ol>

Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley: The ML test score: A rubric for ML production readiness and technical debt reduction. BigData 2017: 1123-1132

<ol start="9">

 <li>(Extra credit) Please go through the excellent talk given by Kate Crawford at NIPS-2017 Conference on the topic of “Bias in Data Analysis” and write a brief summary (at most one page) of the main points.</li>

</ol>

Kate Crawford: The Trouble with Bias. Invited Talk at the NIPS Conference, 2017. Video: <a href="https://www.youtube.com/watch?v=fMym_BKWQzk">https://www.youtube.com/watch?v=fMym_BKWQzk</a>

<ol start="10">

 <li>(Extra credit) Please go through the following program on societal impacts of AI and write a brief summary (at most one page) of the main points.</li>

</ol>

Video: <a href="https://www.pbs.org/wgbh/frontline/film/in-the-age-of-ai/">https://www.pbs.org/wgbh/frontline/film/in-the-age-of-ai/</a>


