import numpy as np
import random
from random import choices
import matplotlib.pyplot as plt


#success probability for a measurement after k iterations 
def win_probability_for_k_iterations(t,N,k):
    theta = np.arcsin(np.sqrt(t/N))
    p_win = (np.sin((2*k + 1)*theta))**2

    return p_win

#Monte Caro experiment that determines the queries that classical sampling needs
def Q_classical_sampling(t,N):
    queries = 0

    while(True):
        queries += 1

        if random.uniform(0,1) <= t/N:

            return queries

#Monte Caro experiment that determines the queries for the algorithm that assumes t is known in advance
def Q_known_t(t,N):
    queries = 0
    theta = np.arcsin(np.sqrt(t/N))
    k_tilde = np.floor(np.round((np.pi/2-theta)/(2 * theta), 3))
    print(k_tilde)
    p_win = (np.sin((2*k_tilde + 1)*theta))**2

    while(True):
        queries += k_tilde + 1 # k_tilde iterations cause k_tile queries + 1 query for checking if the measurement was correct 

        if random.uniform(0,1) <= p_win:

            return queries

#Monte Caro experiment that determines the optimal queries for the algorithm that assumes t is known in advance
def Q_known_t_optimal(t,N):
    queries = 0
    theta = np.arcsin(np.sqrt(t/N))
    min_cost = 1/win_probability_for_k_iterations(t,N,1)
    k_tilde = 1

    for k in np.arange(1,9.2*np.sqrt(N),1):
        new_cost = k/win_probability_for_k_iterations(t,N,k)

        if new_cost < min_cost:
            k_tilde = k
            min_cost = new_cost

    p_win = (np.sin((2*k_tilde + 1)*theta))**2

    while(True):
        queries += k_tilde + 1 # k_tilde iterations cause k_tile queries + 1 query for checking if the measurement was correct 
        if random.uniform(0,1) <= p_win:

            return queries
        
#Monte Caro experiment that determines the optimal queries for the algorithm that assumes t is known in advance without flooring iterations k
def Q_known_t_optimal_no_flooring(t,N):
    queries = 0
    theta = np.arcsin(np.sqrt(t/N))
    k_tilde = (np.pi/2-theta)/(2 * theta)
    p_win = (np.sin((2*k_tilde + 1)*theta))**2

    while(True):
        queries += k_tilde + 1 # k_tilde iterations cause k_tile queries + 1 query for checking if the measurement was correct 
        if random.uniform(0,1) <= p_win:

            return queries
        
#Monte Caro experiment that determines the queries for the algorithm by Boyer et al.
def Q_unknown_t(t, N,m=6/5, λ=6/5):
    θ = np.arcsin(np.sqrt(t/N))
    m_new = m
    k_iterations = 0
    queries = 0

    while k_iterations <= np.sqrt(N): #timeout after sqrt(N) steps, where we would have expected a solution to be found already => no solution is there eg t=0
        j = random.randint(0,np.floor(m_new-1)) #chooses a j randomly between 0 and m-1
        p_win_step = (np.sin((2*j + 1)*θ))**2 #win probability for j iterations
        k_iterations += j
        queries += j + 1

        if random.uniform(0,1) <= p_win_step:
            return queries
        else:
             #adding j queries for each iteration of the algorithm
            m_new = min(λ*m_new, np.sqrt(N))
    else:

        return queries

#Monte Caro experiment that determines the queries for the modified algorithm, that takes into account that t is bounde by some t_max
def Q_unknown_t_for_uniform_on_intervall(t, N, prior, λ=6/5):
    θ = np.arcsin(np.sqrt(t/N))
    min_cost = 1/win_probability_for_k_iterations(max(prior)+1,N,1)
    m_new = 1

    for k in np.arange(1,9.2*np.sqrt(N),1):
        new_cost = k/win_probability_for_k_iterations(max(prior)+1,N,1)

        if new_cost < min_cost:
            m_new = k
            min_cost = new_cost

    k_iterations = 0
    queries = 0

    while k_iterations <= np.sqrt(N): #timeout after sqrt(N) steps, where we would have expected a solution to be found already => no solution is there eg t=0
        j = random.randint(0,np.floor(m_new-1)) #chooses a j randomly between 0 and m-1
        p_win_step = (np.sin((2*j + 1)*θ))**2 #win probability for j iterations
        k_iterations += j
        queries += j + 1

        if random.uniform(0,1) <= p_win_step:
            return queries
        else:
             #adding j queries for each iteration of the algorithm
            m_new = min(λ*m_new, np.sqrt(N))    
    else:

        return queries
    
#work out the expected aueries by taking the average over multiple monte carlo experiments
def expected_queries(t, N, method, n_sample_size, **initialisation):
    iterations = np.sum(np.array([method(t, N, **initialisation) for i in np.arange(0,n_sample_size, 1)]))/n_sample_size
    print(iterations)
    return iterations

#takes a function and returns array with weights for t=1 to t=N
def distribution(function, N, **kwargs):
    dist_args = function.__code__.co_varnames[:function.__code__.co_argcount]
    dist_params = {key: value for key, value in kwargs.items() if key in dist_args}
    dist = np.array([function(xi, **dist_params) for xi in np.arange(1,N+1)]) #Don't let t=0 happen
    dist = dist/np.sum(dist)
    return dist

# Average search iterations for given distribution, with weighted sampling out of distribution
def average_search_iterations_for_given_distribution(distribution, N, method, n_sample_size=100, m=6/5, λ=6/5, **method_params):
    k_average_tot = 0.0

    for i in range(0,n_sample_size):
        t_choice = choices(np.arange(len(distribution)), distribution)[0]
        #print('t is chosen to be', t_choice)
        
        k_average_tot += method(t_choice, N, **method_params)
        #print(k_average_tot)
    k_average = k_average_tot/n_sample_size
    return k_average      


#Functions that plots distribuions and works out the average number of iterations for each method 
def average_search_iterations_for_given_distribution_and_methods_plot(distribution, N, methods, n_sample_size, xlim=100, **kwargs):
    x = np.arange(len(distribution))
    y = distribution
    plt.plot(x,y)
    plt.plot(x,y)
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.xlim(0,xlim)
    for meth in methods:
        method_args = meth.__code__.co_varnames[:meth.__code__.co_argcount]
        method_params = ({key: value for key, value in kwargs.items() if key in method_args})
        result = average_search_iterations_for_given_distribution(distribution, N, meth, n_sample_size, **method_params)
        print("Average Iterations with method", meth.__name__, ":", result)


#Bayes framework

#Bayesian update process
def Bayes_Update(k, prior):

    #work out the likelihood p(k|t) for each possible t
    likelihood = np.array([1-win_probability_for_k_iterations(i,len(prior)-1,k) for i in np.arange(len(prior))])

    #work out marginal likelihood (normalisation)
    marginal_likeliood = np.dot(likelihood, prior)

    #calculating p(k|t)p(t)
    numerator = likelihood * prior

    return numerator/marginal_likeliood

#decision rule 1 "highest success probability for next measurement"
def decision_one_more_shot(prior):
    nonzero_indicies = np.nonzero(prior)[0]

   #Don't allow k=0 since min(...) woud always be 0 then. If k=0 is optimal, classical sampling would do the trick
    max_cost = np.array([win_probability_for_k_iterations(i, len(prior), 1) for i in nonzero_indicies +1])
    max_value = np.dot(max_cost, prior[nonzero_indicies])
    k_opt = 1
    
    for k in np.arange(1,3*np.sqrt(N)+1, 1):
        cost = np.array([win_probability_for_k_iterations(i, len(prior), k) for i in nonzero_indicies +1])
        max_new = np.dot(cost, prior[nonzero_indicies])

        if max_new > max_value:
            k_opt = k
            #print(k_opt)
            max_value = max_new

    return k_opt

#decision rule 2 "lowest number of iterations to succes"
def decision_cost(prior):
    nonzero_indicies = np.nonzero(prior)[0]

   #Don't allow k=0 since min(...) woud always be 0 then. If k=0 is optimal, classical sampling would do the trick
    min_cost = 1 * np.array([1/win_probability_for_k_iterations(i, len(prior), 1) for i in nonzero_indicies +1])
    min_value = np.dot(min_cost, prior[nonzero_indicies])
    k_opt = 1
    
    for k in np.arange(1,3*np.sqrt(N)+1, 1):
        cost = k *np.array([1/win_probability_for_k_iterations(i, len(prior), k) for i in nonzero_indicies +1])
        min_new = np.dot(cost, prior[nonzero_indicies])
        #print(minimising_new)
        if min_new < min_value:
            k_opt = k
            #print(k_opt)
            min_value = min_new

    return k_opt

def Bayesian_update_routine(prior, decicion_rule):
    #decide which k optimises the chance at getting a solution in the next step
    
    k = decicion_rule(prior)
    print(k)
    
    #update beliefs
    prior = Bayes_Update(k, prior)

    return prior, k

#Monte Caro experiment that determines the queries needed for the Bayesian algorithm
def Q_Bayes(t, N, prior, decision_rule):
    q = 0

    while True:
        k_decision = decision_rule(prior)
        q += k_decision
        
        #Doing k Grover iterations
        theta = np.arcsin(np.sqrt(t/N))
        p_win = (np.sin((2*k_decision + 1)*theta))**2

        #measuring
        q += 1
        if random.uniform(0,1) <= p_win:
            return q
        else:
            #updating our beliefs according to Bayes rule
            prior = Bayes_Update(k_decision, prior)   


#used for the update plot
def Q_Bayes_with_plots(t, N, prior, decision_rule):
    q = 0

    state_of_distribution = 0

    while True:
        k_decision = decision_rule(prior)
        q += k_decision

        #print(k_decision)
        
        #Doing k Grover iterations
        theta = np.arcsin(np.sqrt(t/N))
        p_win = (np.sin((2*k_decision + 1)*theta))**2

        #measuring
        q += 1
        if random.uniform(0,1) <= p_win:
            return q
        else:
            state_of_distribution +=1
            print('update #', state_of_distribution)
            #updating our beliefs according to Bayes rule
            prior = Bayes_Update(k_decision, prior)
            newlabel = 'updated prior #' +str(state_of_distribution)
            plt.plot(np.arange(len(prior)), prior, label=newlabel)
    