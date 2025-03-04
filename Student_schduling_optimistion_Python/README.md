The report addresses the issue of student-level scheduling optimization in a hybrid learning environment, particularly in response to the COVID-19 pandemic. The goal is to design a scheduling system that assigns students to in-person classes while adhering to social distancing guidelines. The problem is framed as an optimization challenge that minimizes student excess in classrooms, ensures equitable attendance distribution, and maximizes classroom utilization.

Mathematical Formulation:
Defined parameters (students, classes, timetable, capacities).
Established an objective function to minimize total excess (TE), surplus simultaneous excess (SSE), and total deviation (TD).
Decision variables included student-group assignment, excess students per class, and auxiliary variables for optimization.

Constraints Implemented:
Ensured each student is assigned to one group.
Maintained classroom capacity limits.
Balanced student distribution across groups.

Optimization Method:
Used Gurobi solver in Pyomo (Python) for Mixed-Integer Programming (MIP) optimization.
Weighed different factors to find an optimal number of student groups.
Explored different social distancing constraints (50% and 33% capacity).
Data Structuring:

Used three key data frames: Student-Class Enrollment, Room Capacity, and Class Timetable.

Conclusion
1. The optimization successfully minimized total excess (TE) and surplus excess (SSE), with most cases achieving an optimal solution in under 10 seconds.
2. Increasing the number of groups reduced excess but increased total deviation (TD).
3. The study suggests future improvements, such as:
a. Integrating machine learning (e.g., genetic algorithms) for better predictions.
b. Allowing multiple student groups per day to improve social interactions.
c. Offering unfilled classroom seats to students on a first-come-first-serve basis.