
TODO:

1. <s>Assign (m1, m2) masses, put both stars into the single systems' pool, and  *then* estimate the single system's mass.</s>
3. <s>Find the relation between G and the distance to the MS with secondary binary mass</s>
5. <s>Split the code into binary probabilities assignment and mass estimation?</s>

2. Use ASteCA's binary probability to identify binaries
4. Decide what to do with the "dip" in the G=[15, 17] range
6. Use LOWESS method to estimate the envelope?
7. Define a binary ridge line using m1=m2 and assign binary probabilities proportional to the distance to each ridge line/isochrone. This would remove the necessity for the `thresh_binar` parameter (which is rather arbitrary)
8. Use *only* the envelope method to estimate binarity. Combined with 7., this would take care of 4.


## Description of the code

```
 run
    |--> readData
        |--> readINI
    |--> readData
        |--> loadClust
    |--> totalMass
       |--> IMF_CDF
    |--> binary
        |--> ID
            |--> clustHandle
                |--> splitEnv
                    |--> isochHandle
                        |--> interp
            |--> isochHandle
                |--> isochProcess
                    |--> readData
                        |--> loadIsoch
                    |--> move
            |--> clustHandle
                |--> split
            |--> clustHandle
                |--> singleMasses
    |--> binary
        |--> masses
            |--> isochHandle
                |--> isochProcess
                    |--> readData
                        |--> loadIsoch
                    |--> move
            |--> clustHandle
                |--> binarMasses
    |--> totMass
        |--> totalMass
            |--> get
                |--> getIMF
                |--> prepObsMass
                |--> tremmel
        |--> totalMass
            |--> get
                |--> getIMF
                |--> prepObsMass
                |--> tremmel
        |--> totalMass
            |--> extrapolate
```

### Generate the IMF's inverse CDF

Generate the inverse CDF for the selected IMF. This is done once at the beginning of the code, and allows the random sampling of the IMF later on.


### Split single/binary systems

Assign probabilities for each star of being a single or binary system. Two methods are currently supported:

* Envelope: 


