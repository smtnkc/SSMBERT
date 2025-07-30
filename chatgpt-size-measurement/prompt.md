To perform software size measurement based on natural language requirements, there are two objective measurement techniques tailored for different software architectures: COSMIC Function Points (CFP) and MicroM. Each method provides a unique abstraction level for interpreting software size and supports applicability across varied requirement styles and architectural designs.

---

**COSMIC**, as a Functional Size Measurement (FSM) method, quantifies functional user requirements by counting data movements: Entry (E), Read (R), Write (W), and Exit (X). Each of these movements is assigned a unit value, contributing to the overall size:

`CFP = E + R + W + X`

COSMIC has been widely adopted in industry due to its transparency and architecture independence and is particularly suited for transactional systems.

---

**MicroM**, on the other hand, is specifically designed to address the limitations of traditional FSM methods in modern software architectures such as microservice-based systems. Instead of focusing on data movements, MicroM measures behavioral characteristics by counting events categorized into three levels: Interaction (I), Communication (C), and Process (P):

`MicroM = I + C + P`

MicroM supports size measurement at multiple abstraction levels (functional, architectural, and algorithmic) making it particularly effective when applied to loosely structured or high-level requirement artifacts.

---

Using the background information provided, compute the COSMIC and MicroM measurements for each use-case in the attached CSV file. Specifically, for each row, calculate and append the following columns with their respective values: `Entry`, `Read`, `Write`, `Exit`, `CFP`, `Interaction`, `Communication`, `Process`, and `MicroM`.
