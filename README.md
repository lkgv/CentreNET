# CentreNET

A center of mass based, nonproposal architecture for instance segmentation.

---
The output of network is an offset map with size M\*N\*2, indicating the offset from center of mass of the object that the pixel belongs to current location for each point.

---
**TODO**
* normalization for offsets when training
* replace configparse with YAML
* change network structure
