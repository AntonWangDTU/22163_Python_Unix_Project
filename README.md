K-means clustering:
The main goal of this project was to build a Python script with the ability to run the K-means algorithm.
Furthermore, we wanted to improve the algorithm with the K-means++ method for picking initial centroids.


About the kmer class:


Class kmer():

  intieres med:
    -filnavn
    - Antal clusters


  Load method:
    This method loads the data from a lst file and saves the ID and the Data(as nunpy array) 

  _eucledian distance method:
    Method for internal use - calculates the eucledian distance between two vectors( a row and a centroid) 

  _pick centroids at random 
    pick initial random centroids 

  A method for kmeans ++

  Cluster method:

    1. Uses a method for picking inital centroids 
    2. Assigns the data points to the centroids
    3. Calculates the new centroids
    4. repeats the above for a limited amount of iterations or until convergence 

  Write method:
    writes to the terminal
    
