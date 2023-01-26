ax = plt.axes(projection='3d')
ax.scatter(contact_x,contact_y, U_exact_contact - U_pred_contact,cmap='viridis', edgecolor='none');
plt.show()

ax.plot_trisurf(contact_x,contact_y, U_pred_contact,cmap='viridis', edgecolor='none');


U_inner_pred_np = U_inner_pred.cpu().detach().numpy().squeeze()
ax = fig.add_subplot(3, 2, 1, projection='3d')
ax.plot_trisurf(input_inner[:,0],input_inner[:,1], U_inner_pred_np,cmap='viridis', edgecolor='none');


