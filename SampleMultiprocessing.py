import multiprocessing as mp
import time

# Another way of using multiprocessing

def burnTime(t, k):
	print "Entering Process %d" % k
	time.sleep(t)
	print "Exiting Process %d" % k

if __name__ == '__main__':
	pList = []
	for k in range(0,6):
		# Create process k (but do not start)
		pList.append(mp.Process(target=burnTime, args=(0.25, k)));

	for k in range(0,6):
		# Start process k
		pList[k].start()

	for k in range(0,6):
		# Wait for process k to finish
		pList[k].join()
