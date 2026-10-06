# Task recipes

```bash
# T1
ve task1 --inputs E8.5_RNA.h5ad E9.5_RNA.h5ad --times 8.5 9.5 \
  --target-time 10.5 --knowledge knowledge/v1 --output t1_E10.5.h5ad

# T2 interpolation
ve task2 --inputs embryo_E7.25.h5ad embryo_E8.0.h5ad --times 7.25 8.0 \
  --target-time 7.5 --knowledge knowledge/v1 --output t2_E7.5.h5ad

# T3
ve task3 --wt-train WT_E9.5.h5ad --ko-train Mab21l2_KO_E9.5.h5ad \
  --wt-target WT_E8.75.h5ad --source-gene Mab21l2 --target-gene Gata4 \
  --knowledge knowledge/v1 --output t3_Gata4.h5ad
```

T1 requires expression and exact board-ordered genes. T2 and T3 additionally require `obsm['spatial_3D']`. Cell labels are used only from training inputs; submissions contain expression and, where required, coordinates.
