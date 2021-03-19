# Montreal-Crime-Analytics-using-Heuristic-Algorithms

Here are the list of libraries used by the project:
  1. shapefile
  2. matplotlib.pyplot
  3. numpy
  4. collections
  5. matplotlib.patches

So before to run the program, make sure to install them properly.


INSTRUCTIONS OF RUNNING AND TESTS.

Firstly, open the entire file using PyCharm. Then simply run the file 'main.py'.

Then, in the python run command line, it will promot user to input a integer n as the size of grid. By default(press enter directily) the size will 20 which means the map will be shaped to be a 20 by 20 grids evenly.

Then, the command line will promot to input a threshold with range 0 to 1. By default(press enter directily) the threshold is 0.5.

After this, the command line will shows the threshold value, the average, standard deviation of all grids, and the map of total crimes in each grid, here is a output example (n = 20, threshold = 0.5):
	The threshold value =  37
	The average of all grids =  47.525
	The standard deviation of all grids =  49.1031789240893

	Display the number of total crimes in each grid: 
 	[[ 38  32  34  24   0  82   5  41   7  36   0  16   0  51  49  25  24   3   15  35]
 	[ 47  71  12  24  62  24  51   2  76   8  21   0   9  62  31  84   9  11     3  29]
 	[ 38  66  47  64  52  41   9 113   9   0   0   0  39   0  27  72  30  14    24  25]
 	[ 29  16  31  73 102  25  50  34  41  49  35  15  17  21  66  67  44  31    38  91]
 	[ 57  78  70  48  54  95  18  75  28  70  38  47  60  88  21  66  44  70    75  23]
 	[ 23  84  53  23  83  23  71  12 139  30  62 114   0  65  76  56 119 100    47  21]
 	[ 30   0  32  90   0 150  76  45  38  57  47  56  40  32 109 255  88  98    39   0]
 	[  0   0   0   0  39  61  70 205  73 140  58  61  94  61 212 222 105  50    41  20]
 	[  0   0   0   5   0   0  21   2 132 100 182 104   0  62 135  42 167  80    14  19]
 	[  0   0   0   0   0  15   9  55  70  71  55 117  65  63  59  81   4  40    77  46]
	[  0   0   0   0  55  22 131  93  30 166  56  19  76  79  78  37  24  23    59  98]
 	[  0   0   0   0   8  10  44 149 111 134  90  64  44  57  61  79 270 218    67   6]
 	[  0   0   0   0  23  15   0  25  30  98 157  60  97  39  40 162  27  91    71   0]
 	[  0   0   1  35   5  10   0  55  25   0 200  81  31  53  42  74 127 156     5   0]
	[  0   0 187  11  52  45 112 191 103 189  18  25  67  32  65  79 110  79    59   0]
 	[  0   0  37  29  18  58 278  56  95  99   3  52   9  27   2  43  65  25    38   0]
 	[  4   3  11  11 119 165  53  85  46  77  19  11  12  19  16   9   6   6    37   0]
 	[  5   0   0  65 158 153  83  73 123  46  28  24   0  21  11  33   5   2    10   0]
 	[ 17   4  28  81 101 101  46  25  44  12  41   6  27   8   7  19  40   0     1   0]
 	[ 25   6  47  94 110  29  14  16   5   6  10  43   6  18   9  44   5   0     0   5]]


Now, the command line will promot user to imput a start and goal coordinates. Press enter will choose a start and goal point randomly from grid coordinats list. If the points are located inside one grid, then its left bottom coordinate will be chosen. Here are some test data for those points that are not positioned at grid coordinats perfectly.
	test data:
	start point: -73.585, 45.495
	goal point: -73.555, 45.525

After these, the graph with the path from start to goal will show up.
Besides, a total cost of the path and executing time will be displayed in command line.

Finally, a close message shows.
