#!/usr/bin/python


def outlierCleaner(predictions, ages_train, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    import math

    sortedError = []
    for index in range(len(predictions)):
        sortedError.extend(abs(predictions[index] - net_worths[index]))

    sortedError.sort()

    Errors81 = int(math.floor(len(sortedError) * 0.9))

    Errors9 = sortedError[Errors81:]

    for index in range(len(predictions)):
        error = abs(predictions[index] - net_worths[index])
        if error in Errors9:
            pass
        else:
            cleaned_data.append([ages_train[index], net_worths[index], error])
     
    
    
    return cleaned_data

