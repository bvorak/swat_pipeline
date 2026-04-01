
[[TFM Discussion]]
# **Results**

###             This chapter evaluates the propagation of uncertainty in point- and diffuse-source inputs through SWAT simulations and assesses whether the resulting prediction envelopes are consistent with observed water-quality signals under defined data and model constraints.

## **1. Specfication of point-sources in SWAT**

###             The network representation of the Cubillas watershed, as defined in the configuration (.fig) file, is shown in Figure X, including subbasins (green), reaches (red), and point sources (orange). Point sources are not part of the default SWAT configuration and must instead be defined by the user via external input files, which contain time-series of flow rates and water constituent concentrations. Even if a given subbasin has no point sources, a corresponding file must still be specified. All elements, including intermediate aggregates (blue), are encoded using a Hydrological Storage Number (HYD), which defines the topological connectivity of the system from upstream to downstream. Note that one point-source file is defined for each subbasin, with each subbasin also associated with a corresponding reach.

FIGURE X.  [[U1]](#_msocom_1) Network (python) visualization of the Cubillas .fig config of Subbasins (green), Reaches (red) and manually inserted POINT SOURCES (orange). All elements and intermediate aggregates (blue) are encoded via a Hydrological Storage Number (HYD)  CAN YOU BE MORE SPECIFICA: WHAT IS HYDROLOGICAL STORAGE NUMBER . WHAT ARE THE INTERMEDIATE AGGREGATES.

**2. Interpolated soil nutrient concentration fields**

Here you present your results of the interpolation exercise!! And say something about them (minimum values etc)!!! Also you could mention the fact that REDIAM and LUCAS do not match, which could be due to differences in methodology (???) or high variability of the field you are trying to interpolate! And you state what are the magnitude of the errors!!! Variability and errors! This should be the outcome of this section.

**3.** **Event versus non-event conditions.**

 The scientific literature consistently shows that nutrient loads and concentrations in rivers reach their highest values during high-flow (storm) events (e.g. Kelly et al. 2019, Zhang et al. 2024). This has been attributed to enhanced hydrological connectivity and the activation of rapid transport pathways during storm events, promoting nitrogen losses via leaching and phosphorus losses via erosion and sediment mobilization. Concentrations generally peak on the rising limb or near peak flow, with phosphorus showing an early, sharp peak due to erosion and sediment mobilization, and nitrogen exhibiting more variable timing depending on subsurface transport contributions (Bowes et al. 2005; Bieroza and Heathwaite 2015). The experimental dataset was subdivided into two subsets: one representing baseflow (low-flow) conditions and the other corresponding to high-flow (storm event) conditions. This subdivision was applied to account for flow-dependent differences in the relationships between variables. Storm events (high-flow conditions) here were defined as periods of at least two consecutive days[[U2]](#_msocom_2)  during which flow exceeds the 75th [[U3]](#_msocom_3) percentile of simulated flows, while all other periods are classified as baseflow (low-flow) conditions. In general, available field data are biased and predominantly represent baseflow conditions. For example, almost 2/3 of the available water quality data for TN correspond to baseflow conditions, the remaining representing observations under high-flows (not necessary peak-flows). For P, XXX. NOTE: I THINK THERE IS NO NEED TO INCLUDE A FIGURE TO SAY THIS. OR YOU MAY WANT TO INCLUDE A FIGURE, BUT THE FIGURE NEEDS TO SAY SOMETHING … FOR EXAMPLE, DO WE SEE AT ALL THE HISTHERESIS IN CONCENRATION DURING STORMS IN SWAT RESULTS THAT THE LITERATURE MENTIONS? …

+ Kelly et al. 2019. DOI: [10.1021/acs.est.8b05152](https://doi.org/10.1021/acs.est.8b05152)

+ Zhang et al. 2024 [https://doi.org/10.1016/j.envres.2024.119762](https://doi.org/10.1016/j.envres.2024.119762 "Persistent link using digital object identifier")

+ Bowes et al. 2005. doi:10.1016/j.watres.2004.11.027

+ M.Z. Bieroza and , A.L. Heathwaite. 2015. doi: 10.1016/j.jhydrol.2015.02.036

.

12 TN measurements during non-event conditions. 6 TN measurments during event conditions (la escala … podriamos modificarla para que se viera mejor los números[[KW4]](#_msocom_4) )– y no sé que quieres decir de estas gráficas?? No se hasta qué punto esto hay que presentarlo. Ya lo estás diciendo arriba EN EL TEXTO!!? En todo caso no hacen falta dos figuras!!! Porque puedes representar high-flow concentration records con un simbolo (e.g. diamods) y usas otro símbolo (e.g. cicles) to represent concentrations under low-flow conditions. Also I am not sure if you should represent above those figures (as subplots) the magnitude of the simulatd and observed flow rates. Do high-flow and low-flow periods coincide? I can look at that (but I will need the final simulations) – HOW ABOUT DOIN THIS FOR P TOO??[[KW5]](#_msocom_5) 

## **4.** **Dominant pollution sources: point vs. non-point**  

The relative influence of point-source and diffuse nutrient inputs on simulated nutrient concentrations upstream of the Cubillas Reservoir was quantified using a structured perturbation analysis. In this framework, point-source discharges and diffuse (nonpoint) inputs were systematically varied between their minimum and maximum values (input bounds). Three minimum-maximum scenarios were considered. First, point-source inputs were varied between their minimum and maximum bounds while diffuse inputs were held constant. This will be referred to as the W (for wastewater) scenario. In the second one (referred to as S – for soil - scenario), diffuse inputs were varied while point sources remained fixed. Finally, in the third scenario (W+S), both point- and diffuse-source inputs were simultaneously varied across their respective ranges. For each scenario, simulations conducted with minimum and maximum input conditions resulted in prediction envelopes representing the range of model outputs under input uncertainty. The relative importance of point and diffuse sources, as well as their combined effects, on model response (measured in terms loads or concentrations) was assessed by comparing the width and divergence of those envelopes for total phosphorus (TP) and total nitrogen (TN) among scenarios. Differences in envelope extent (or spread) among scenarios were used to infer source dominance and potential interactions between point and diffuse inputs. In simple terms, the comparison between the three scenarios tells us where most of the influence on P or N concentration in the model is coming from, i.e.

- **W scenario** (point sources only): If the range of nutrient concentrations (the envelope) were wide in this scenario, it would be an indication that point sources (wastewater discharges) strongly controls P/N concentrations
- **S scenario** (diffuse sources only): If the range is wide in this case, it means that runoff from land (e.g., agriculture, soil erosion) is the main driver of nutrients concentrations in the river and loads into the reservoir.
- **W+S scenario** (both together): Comparing the results of this scenario to the other two, will allow us to identifiy the combined effect. If it is much wider than both, both sources are important and may reinforce each other. If it is similar to just one scenario, that source is dominant. If it is only slightly larger than the sum of both sources, the sources act independently which suggests that there is no strong interaction.

            In summary, by comparing the spread of nutrient predictions in each scenario, one can tell whether those nutrients in the system are mainly controlled by point sources, diffuse sources, or a combination of both—and whether their effects interact or not. Average yearly TN loads and the corresponding min–max envelopes under the W+S scenario are shown in Figure 9. In this case, the uncertainty envelopes were calculated as the average value plus or minus the absolute median of the relative interpolation error. In Figure 10, the envelope is instead defined using the 75th percentile of the relative soil interpolations..

Figure 9?? Scenario 204[[U6]](#_msocom_6) [[KW7]](#_msocom_7) , W+S, using absolute median of relative error for SOIL input interpolation. QUESTIONS – what is scenario 204 (we need to be clear with the nomenclature!!). What is the magnitude of the relative error for point sources?? Also 50% --- If this is the case, then we should continue using the median! Qué representas en la gráfica de abajo?? No tine leyenda ni escalas!![[KW8]](#_msocom_8) 

Figure 10: TN kg/year accumulated kg/day predictions; Scenario 201 (S + P[[U9]](#_msocom_9) ), using P75 of relative error for SOIL input interpolation. Question – what is the last gray bar??

Figure 10 showing yearly summed kg/day loads of TN for the most permissive S+P scenario, the large min-max envelope showing high model sensitivity. Notably the min realization of this scenario with very large uncertainty envelope ( which for TKN SOIL interpolation basically perturbates the inputs between 0 and twice the interpolated values from REDIAM measurements) predicts even less load than the BASE scenario (which just uses SWAT default loads). Indicating that SWAT defaults already consider some SOIL Nitrogen initialization above 0, probably estimated from total Carbon values specified in the .sol files.

Figure 11: zoom in on TN kg/week accumulated kg/day predictions; Scenario 201 (S + P), using P75 of relative error for SOIL input interpolation

 Figure 11 shows in detail how much more sensitive the model is during high load events than during low load events. QUESTION – what do we mean with more sensitive?? We mean that the uncertainty range increases during flow events? If we are using the relative magnitude of the range as a measure of uncertainty [[KW10]](#_msocom_10) … can you give specific numbers?? For example, on 1999-1 during perak inflow rates …. (you are saying below that you use “_the relative spread as a comparable measurement of how sensitive the model is to each input load scenario_”!)

### 4.1. **Width of uncertainty** **envelopes**

            Envelope width is summarized by the relative spread (min–max divided by the median) for all days and for event/non-event subsets. This gives a comparable measure of how sensitive the model is to each input load scenario, for both TP and TN. CAN YOU BE MORE EXPLICIT ON HOW YOU DO THE CALCULATION? FOR EACH DAY OF THE SIMULATION YOU CALCULATE MIN-MAX / MEDIAN AND THEN YOU AVERAGE FOR THE WHOLE TIME SERIES?? It is NOT CLEAR AT ALL TO ME WHAT CALCULATIONS ARE YOU MAKING? O

Figure 12: Relative min-max envelope widths of simulated TN and TP loads at the outlet of Reach 13. Comparing the three scenarios  (Waste, Soil and W+S)  for all days vs. flow-event and non-flow-event days. Can we put TN and TP in separate subplots?? In this manner we are directly comparing scenarios in a given plot Just a suggestion- if this is work for you we leave it as it is.

### As shown in Figure 12, TN responded strongly to pure SOIL input perturbations (174%), nearly double the response to pure WASTE perturbations (89%). In contrast, TP responses were comparable for SOIL (140%) and WASTE (154%) perturbations[[U11]](#_msocom_11) . For the combined WASTE+SOIL scenario, responses closely reflected those of the dominant input type (TN: 171%; TP: 157%). This pattern is also observed under baseflow conditions (non-event days). During event days (above the 75th percentile), however, the model responds more strongly to SOIL perturbations in both TN (SOIL: 71%; WASTE: 39%) and TP (SOIL: 78%; WASTE: 59%).

### **4.2. Shifts relative to** **baseline simulations**

                Model results were shifted in all scenarios relative to the BASE scenario —which assumes no added point sources and evaluates diffuse input loads using the default initial N/P content in soil—. The position of the uncertainty envelopes (or bias) with respect to the BASE prediction was quantified as follows. For each day of the simulation, the mean N and P loads of the uncertainty envelope were calculated as the average of the loads obtained using the minimum and maximum values of point-source inputs and soil nutrient concentrations within their defined uncertainty ranges. The daily ratio of this mean load to the nutrient load predicted in the BASE scenario was then computed. The ratio of this mean load to the nutrient load predicted in the BASE scenario was then computed on a daily basis. The median or mean magnitude of this ratio over the simulation period is shown in Figure X[[U12]](#_msocom_12) . A positive bias indicates that the corresponding summary metric (mean or median) of the uncertainty envelopes lies above the BASE prediction. Its magnitude reflects the contribution of added point-source loads, as well as the generally higher observed soil nutrient concentrations compared to the default initial values obtained through the standard initialization procedures.

Figure 11: BIAS introduced under three load scenarios (WASTE, SOIL, and W+S) compared to the BASE scenario,; TN = 18; TP =25[[U13]](#_msocom_13) 

            The BASE model predicts substantially higher loads for TN (median, p50: 865 kg/day) than for TP (p50: 3.24 kg/day). While the absolute shifts differ markedly between nutrients, the shifts normalized by their respective median values in the BASE model are very similar (TN: 0.79; TP: 0.76).[[U14]](#_msocom_14) 

##             Biases for TN are approximately one order of magnitude larger than those for TP, regardless of the scenario. For both nutrients, larger biases occur in the SOIL scenario than in the WASTE scenario.

            The mean bias in the SOIL scenario relative to the BASE scenario varies considerably between events and base-flow periods. Total phosphorus (TP) loads exhibit large increases during events (+1424) compared to base-flow periods[[U15]](#_msocom_15)  (+33). Total nitrogen (TN) loads have a similar pattern. The bias, in general, is much higher during events (+2262) than during base-flow periods (+1083). In contrast, the WASTE scenario produces more consistent shifts in nutrient concentrations across flow regimes. TP changes are similar between events (+36) and base-flow periods (+31), while TN shows moderate increases in both events (+216) and base-flow periods (+171).

Figure 12. FIGURE CAPTION!! AS ABOVE I WOULD SEPARATE N & P. THEIR CONCENTRATION RANGES/LOADS ARE AT LEAST ONE ORDER OF MAGNITUDE!!! HENCE, WE DO NOT SEE ALMOST ANY DIFFERENCE IN THE CASE OF P.

            Concentrations and total loads can tell different stories. When examining nutrient concentrations predicted under the different scenarios considered, the shifts relative to the BASE scenario generally follow the same pattern as total nutrient loads. The exception is TN concentration in the SOIL scenario, which shows a larger positive bias under baseflow (non-events) conditions (+19%) compared to storm or highflow conditions (+7%). This suggests that (1) the contribution of non-point sources to TN under baseflow conditions are disproportionately important for concentrations; (2)  event-driven mobilization of soil nitrogen plays a smaller role in determining TN concentrations. The persistent “background” TN leaching or slow flow pathways (lateral flow) contributes more to concentrations than the sudden pulses during storms. Total TN loads, however, still depend heavily on event-driven transport, because storms move large quantities of nitrogen, even if they don’t proportionally increase concentrations. NOT SURE WHAT IS A PLAUSIBLE EXPLANATION IN TERMS OF PROCESSES (THIS WILL REQUIRE SOME WORK THAT RUBEN OR ME WILL HAVE TO DO)

            The bias calculated for the W+S scenario is similar in magnitude to the sum of biases estimated in the SOIL and WASTE scenarios. For example, TP predictions are, on average, 33 mgP/L higher in the WASTE simulations compared to the BASE scenario, 416 mgP/L higher in SOIL, and 445 mgP/L higher in the combined W+S scenario. This occurs independently of the flow conditions (event vs. baseflow conditions). Similar trends are observed for TN.  These results suggest that point and non-point sources act contribute independently to river nutrient concentrations, with no evidence of strong interactions.

## **5.- Envelope coverage of validation data**

### Validation data sub-sets vs. Scenarios

**Figure 13:** Example from the visual exploration workflow, illustrating how scenario envelopes and observations are inspected across the whole simulation period. Figure 14, being a zoom in on three above detection limit TN validation data points.

Figure 12: Dashbaord visualization of model output compared to measurements for TN; S+W scenario. You do not differentiate event vs. baseflow conditions? I insiste that we probbably need flow rates plotted on top.

Figure 13:: Dashboard visualization of Model output compared to measurements for TP; S+W scenario

[[KW16]](#_msocom_16) NOTA – no veo absolutamente NADA!!! La escala!!!

Overall Model–Observation Fit

This section examines the relationship between simulated loads for reach 13 and measured concentrations at the same reach[[U17]](#_msocom_17) . ~~Conventional goodness-of-fit indicators applied to the raw measurement set did not yield interpretable values due to the sparse and heterogeneous sampling. Consequently, model–data agreement is described using envelope coverage and bias diagnostics, evaluated under two measurement subsets.~~

### Conventional goodness-of-fit indicators applied to the raw measurements were not interpretable due to sparse and heterogeneous sampling. To better capture model–data agreement under these conditions, we use envelope coverage and bias diagnostics, which account for variability in the observations. These metrics are evaluated across two measurement subsets[[U18]](#_msocom_18) . Detection limit for P??? will be set to 0.25 mg/L [[U19]](#_msocom_19) . and for N??[[U20]](#_msocom_20) 

###             For TP, WASTE input perturbations produce a simulation envelope encompassing 68% of the 25 measured values, whereas SOIL perturbations alone capture only 12% of the observations (see coverage fraction in Fig. X). The combined WASTE+SOIL scenario yields a modest improvement over WASTE alone, covering 72% of the measured TP values. For TN, the pattern is reversed: SOIL captures 50% of the 18 observations, WASTE 11%, and WASTE+SOIL 56%.

###             Simulations under the WASTE scenario produce TP estimates that are, on average, 60% higher than observations. SOIL simulations overestimate TP by 108%, and the combined WASTE+SOIL scenario by 226% (0.7 mg TP/L[[U21]](#_msocom_21) , see Pbias results in Fig. X). For TN, WASTE simulations tend to underestimate concentrations by 68%, whereas SOIL and WASTE+SOIL simulations both overestimate TN by approximately 56% (13 mg TN/L).

Load Duration Curves (LDCs) are commonly used in hydrology and water quality modelling to  show how pollutant loads or concentrations vary across the full range of streamflow conditions. They plot the percentage of time a given flow is equaled or exceeded (x-axis) against the corresponding pollutant load or [[U22]](#_msocom_22) concentration (y-axis), helping to identify whether a model over- or underestimates pollutant loads (or concentrations) during low, medium, or high flow conditions. Here, LDCs for TN and TP in Cubillas watershed were constructed from the simulation results are shown in Figs. X and Y. Q1: are they still called Load Duration Curves if you plot concentration? Q2: The x-axis is calculated with flow rates??? Or with loads?? Q3: YOU NEED TO EXPLAIN HERE WHAT are the different curves within the same plot??. Experimental loads are shown as diamonds in the LDC cuves. These estimates were calculated from measured TP or TN concentrations, multiplied by the flowrate simulated with SWAT when the sampling ocurred[[U23]](#_msocom_23) . Q: WHY IS THERE SO MUCH NOISE IN THE GREEN LINE?? WHAT DOES IT REPRESENTS?? Under the W+S scenario, TN tend to be overestimated during high load conditions while during low load conditions it rather underestimates TN. A comparable, albeit weaker, pattern is observed for TP. From these plots you can also discuss how many points are within the uncertainty band?? Or not?

 HOW is MEAN LOAD estimated? For each day, you calculate the average between max and min simulations?? And you average (in time) those averages between min-max inputs? The calculation routes is not clear to me at all!!

### When considering only low-flow (baseflow) conditions, the model including WASTE and SOIL perturbatinos (W+S scenario) captures a larger fraction of the observed values, with 75% of TN and 79% of TP measurements falling within the predicted range. SOIL perturbations account for most of the model’s ability to reproduce TN, capturing 75% of observations, but only 5% of TP measurements. In contrast, WASTE perturbations capture 74% of TP observations while representing only 17% of TN. Under baseflow (low-flow) conditions, SOIL strongly overestimates TN, with a PBIAS of 281%, while underpredicting TP by 25%. In contrast, WASTE underestimates TN by 27% and overestimates TP by 31% under the same conditions.

Can we say something about LDCs in this case? OR NOT???

~~If only focus on baseflow conditions, overall coverage of observations increases (TN: 75%, TP: 79%), and SOIL perturbations entail already most of the models TN prediction ability (75%) and almost none for TP (5%), while WASTE conversely describes 74% of TP observations and only 17% of TN obersvations. In non-event conditions SOIL has a even larger positive PBIAS of 281% vs. observed TN values, while it underpredicts TP by 25%. WASTE on the other hand underpredicts TN in non-event conditions by 27% while it overpredicts TP by 31%.~~

### Under high-flow conditions (events), both WASTE and SOIL substantially underestimate TN (by 94% and 84%, respectively) and fail to capture any TN observations within their prediction envelopes. Only the combined WASTE+SOIL scenario captures a small fraction of TN observations (17% coverage). For TP, both WASTE and SOIL overestimate concentrations during event conditions, with biases of 268% and 1151%, respectively. Despite this, WASTE and WASTE+SOIL each capture 50% of TP observations within their envelopes, whereas SOIL captures 33%.

WE NEED TO EXPLAIN CLEARLY OUR CALCULATIONS ---- MAY BE THIS IS SOMETHING TO DO IN THE METHODS, OR ALTERNATIVELY HERE AT THE BEGINNING  

TN = 6

SOIL performs even better in describing measured TN (86% coverage) when we drop the below detection limit values, rather than setting them to 0.25 mg/L. Positive bias of SOIL decreases with this filter to 213% compared to 280% bias of the event prediction without filtering below detection limit values out. (See 2 figures above)

 

---

 [[U1]](#_msoanchor_1)Check fig. numbers!!! They need to be corrlated with the references in the text!!!

 [[U2]](#_msoanchor_2)Confirm please

 [[U3]](#_msoanchor_3)This choice is arbitrary? Can you say why you chose 75%??

 [[KW4]](#_msoanchor_4)No si estamos comparando contra un min-max, como los max son muy altos. Intersante es que el naranja BASE ya es encima de lo medido a veces. Quizas esto es interesante mostrar

 [[KW5]](#_msoanchor_5)Quizas mas facil quitarlas. Querai solo mostrar como se esta filtrando. Los subplots serian posible. Ambos en una grafica, seria ya un cambio del codigo del dashboard.

 [[U6]](#_msoanchor_6)??? What scenario is this? You have not described it!

 [[KW7]](#_msoanchor_7)Ah yes sorry, this is the W+S (absolute-median-of relative error) scenario vs. The 75th percentlile of relative error scenario

 [[KW8]](#_msoanchor_8)Have to check

 [[U9]](#_msoanchor_9)S+P? Scenario 201???

 [[KW10]](#_msoanchor_10)I dont know if we are using the magnitute of the output range as a measurment of uncerteinty. But for this W+S uncerteinty scenario our input-uncerteinty has way more impact in the sense of output variability during events than during non-events.

 [[U11]](#_msoanchor_11)Can you instead specifiy the ranges too? We can do the comparison statistically. If you give me the numbers I can do that for you. We want to say (insead of considerably or slightly different) – wether those differences are ‘statistically significant’. For P, probably they are not!

 [[U12]](#_msoanchor_12)Can you confirm that this is what you do??

 [[U13]](#_msoanchor_13)What is this. How do you use median BASE prediction for TP or TN??

 [[U14]](#_msoanchor_14)Not sure where these numbers come from!! This is for W+S, SOIL, WASTE???

 [[U15]](#_msoanchor_15)This is more formal and technically preferered in hydrology.

 [[KW16]](#_msoanchor_16)update

 [[U17]](#_msoanchor_17)This is what you have done so far – you only used reach 13 to show your results????

 [[U18]](#_msoanchor_18)What is this??? P and N??? or what are these two measurement subsets: baseflow vs. events??

 [[U19]](#_msoanchor_19)(this is taken from where?? Or visually? The lower value in the data set??)

 [[U20]](#_msoanchor_20)This should be stated in material and methods

 [[U21]](#_msoanchor_21)Esto de dónde sale?? You need to guide the reader (ME!!! In this case!!)

 [[U22]](#_msoanchor_22)????

 [[U23]](#_msoanchor_23)Es así? O cómo lo haces??? No lo llego a entender, y no hay ninguna explicación clara sobre tus procedimientos.