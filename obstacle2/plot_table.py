import matplotlib.pyplot as plt
x = [20,50,100,200,500,1000,10000]
x_stick_label = [str(i) for i in x]
y = [0.47,0.35,0.14,0.067,0.035,0.019,0.0042]

plt.xscale('log')
plt.yscale('log')
plt.xticks(x,x_stick_label)
plt.xlabel('number of samples')
plt.ylabel('relative error')
plt.plot(x,y)
plt.show()