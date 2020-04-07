**Problem Statement:** *Develop a solution for addressing traffic problem in the country, the goal is to make roadways efficient by reducing traffic jams and storing traffic data for future development and solutions to develop a traffic free cities.

**Overview of the project:** Road congestion is the prominent cause for traffic which results in extensive wait time and thus results in pollution. The current methods do not compile with the growing needs of a metropolitan city and thus results in such a menace. As these methods do not provide concrete solution’s to the roadways and future planning’s based on population, town planning and other development factors we have come up with a solution that deals with present traffic problems and also helps to design future roadways by using existing data of pollution, population, future town planning such as under construction buildings through crowd source data which will help us to redesign the current roads and thus to curb pollution to a major extent.

**Existing Problem:** The traffic signal system in India does not depend upon the amount of traffic as it is pre defined i.e. The amount of vehicles held up in a signal which varies from 60 sec to 90 second. as a result the clearance of traffic which goes up from large kms suffers as small portion of traffic is released from these time intervals. 
As a result in a four way crossing when one side is suffering from traffic congestions and others are not then too with inequal amount of traffic at each crossing the amount of traffic released is equal which is not appropriate as per the current traffic conditions and also the lane with large amount of traffic has to hold the vehicles till all the other lanes complete which results in more waiting time,  more incoming traffic and thus amounting to more pollution.
But this is not the only problem which results in such humungous traffic jams. Traffic jams are also caused by the amount of people concentrated at only one particular locations such as railway stations, schools, markets…etc. This is caused due to unwanted factors such as hawkers and smaller/no footpaths. A metropolitan city is always growing which means there are under construction buildings which will result in increase in number of people residing in that area. With this information we understand that the amount vehicular movement will increase significantly with the amount of upcoming buildings in that area which the current system lacks in redesigning the roadways which results in traffic congestion.
As a result the problems now vary from static signals to narrow roads to removing unwanted factors from the roads

**Solution:**  In order to solve these problems we have come up with a system that results in

**1.** Every road is constructed with a minimum and maximum amount of vehicles at one time. For eg. Suppose a road xyz is build with a maximum amount of vehicular movement at one time is 100 so if there is 300 vehicles at a time that means there is probability of traffic jams.

**2.** Thus we are creating a map application which will give the shortest path to the users from source to destination but when our application comes to know that it has sent 60 vehicles from the same road and are currently at the same location that means it is 40 vehicles away from the threshold limit. Upon acting to this situation our map will re-route the next users as we have predicted that there might be a traffic situation keeping an offset of the vehicles who are not using the application. This will help us to bifurcate the traffic.

**3.** Bifurcation of traffic is not the only solution as developers we will concentrate upon the offset that are the people who are not using the application.

**4.** In order to avoid this offset i.e.if such a condition arrives that there is a traffic in a particular location we have come up with dynamic traffic light system.
In this system the time taken to release a particular lane will depend upon the amount of traffic in that lane which the data will be received through the cctv cameras and will be processed with the help of machine learning.

**Uniqueness about the idea:**

**1.** Currently all the traffic systems in dia follow static time based working in which every signal is has been given with a certain amount of time to show a particular signal.This is a major cause of traffic as it doesn’t take any other factors into consideration eg.major one i.e.number of vehicles on a particular lane.This leads to inefiiciently control a traffic.
Dynamic signal is a signal that will assess the conditons like number of vehicles, obstructions caused by ongoing contructions and will accordingly set a time for that particular lane.Thus dynamically changing waiting time which will surely positively impact on traffic conditions.

**2.** This above data will be stored in the database and accordingly will show a alternate path on the map which users are using for navigating  which can save them from the hassell of traffic and also prevent from creating more traffic on already traffic jammed locations.

**Software roadmap:**
First with the help of cameras attached to every signals we will do an object detection analysis.With the help of this analysis we will get the number of vehicles and the types of vehicles on that particular lane. This object detection will be done on all the cameras of a particular junctions. Then with the help of this data we will calculate the signal light timer for all signals on the junction .
It would be much more efficient then the static system currently implemented. Based on the number of vehicles and type of vehicles we would calculate the timer of that signal. This data would also be stored on a database. With the help of the data we will show an alternate path to the users using our map for navigation. By doing this we are giving the user an alternate free path without traffic and also stopping the traffic from increasing at a particular spot.


