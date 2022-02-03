import pickle as pk
import sys

def main():
	args = sys.argv[1:]
	f = args[0]
	if f == '-w':
		t = args[1]
		o = args[2]
		with open(t, 'r') as t:
			t = t.read()
			pk.dump(t, open(o, 'wb'))
			print("file wrote !")
	else:
		with open(f, 'rb') as f:
			data=pk.load(f)
			print(data)

main()
