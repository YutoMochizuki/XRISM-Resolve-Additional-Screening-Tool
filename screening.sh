#!/bin/sh
###############################################################
# XRISM Resolve Additional Screening Script
###############################################################

###############################################################
# Set the working directory to the data directory
DATADIR="../fits/"
ANALYSISDIR="../bin/"
###############################################################


###############################################################
# Change to the data directory
cd ${DATADIR}
###############################################################

###############################################################
# Define the observation ID and filter IDs
OBSID="300003010"
FILTERID=(0000 1000 5000)
###############################################################

###############################################################
# Screening
for id in "${FILTERID[@]}"; do

event_file="${DATADIR}/xa${OBSID}rsl_p0px${id}_cl.evt.gz"

echo "${event_file}"

fselect ${event_file} ${DATADIR}/xa${OBSID}rsl_p0px${id}_cl_additional_screening.evt.gz '((((RISE_TIME+0.00075*DERIV_MAX)<58)&&((RISE_TIME+0.00075*DERIV_MAX)>46))||(DERIV_MAX>=15000))&&(((TICK_SHIFT<4)&&(TICK_SHIFT>1))||(DERIV_MAX>=15000)||(DERIV_MAX<6000))&&(((TICK_SHIFT<4)&&(TICK_SHIFT>0))||(DERIV_MAX>=6000)||(DERIV_MAX<2000))&&(((TICK_SHIFT<3)&&(TICK_SHIFT>-1))||(DERIV_MAX>=2000)||(DERIV_MAX<1000))&&(((TICK_SHIFT<2)&&(TICK_SHIFT>-2))||(DERIV_MAX>=1000)||(DERIV_MAX<500))&&(((TICK_SHIFT<1)&&(TICK_SHIFT>-3))||(DERIV_MAX>=500)||(DERIV_MAX<400))&&(((TICK_SHIFT<0)&&(TICK_SHIFT>-4))||(DERIV_MAX>=400)||(DERIV_MAX<300))&&(((TICK_SHIFT<-1)&&(TICK_SHIFT>-5))||(DERIV_MAX>=300)||(DERIV_MAX<200))&&(((TICK_SHIFT<-3)&&(TICK_SHIFT>-7))||(DERIV_MAX>=200)||(DERIV_MAX<100))&&(((TICK_SHIFT<-4)&&(TICK_SHIFT>-8))||(DERIV_MAX>=100))' clobber=yes

done
###############################################################

###############################################################
# Change to the analysis directory
cd ${ANALYSISDIR}
###############################################################
