# [M2 DIPAC] GPU programming (3 ECTS)

This course offers a practical introduction to GPU programming, focusing on the principles and techniques behind parallel computation using the CUDA programming model. Students will explore how modern GPUs execute thousands of threads concurrently, how memory hierarchies affect performance, and how to design efficient kernels for tasks such as matrix computations, scientific computing, and machine-learning-related operations. Through a blend of lectures, hands-on examples, and targeted programming assignments, the course builds core skills in parallel algorithm design and performance optimization, preparing students to create high-performance GPU-accelerated applications and use existing optimized GPU libraries.

# Course organization

We will have seven sessions of 3 hours each (plus mini-exams, c.f. below). The first session is an introduction to the GPU architecture and CUDA programming model. The rest will be a mix of course (around 1h) and lab session (around 2h) on that subject. You might not be able to finish all exercises during the lab session (which is completely normal); you should make sure that you can understand and finish all basic exercises (first two or three) during the session so that you feel more confident to finish the rest on your own.

Please put your name, e-mail address, master's track, and public ssh key here to register for the course:

https://cirrus.universite-paris-saclay.fr/s/3NLLZiJq4kHTYDf

Please also bring your laptop to the course; you will be using it during the lab sessions.

## Session 1: Introduction to GPU architecture and CUDA programming
## Session 2: Introduction to CUDA programming (cont.)
## Session 3: 1D and 2D GPU kernels, matrix multiplication
## Session 4: GPU memory challenges, memory coalescence, matrix multiplication and stencil computations using shared memory
## Session 5: Reduction on GPUs, thread divergence, memory bank conflits, matrix transposition.
## Session 6: GPU streams and running parallel kernels
## Session 7: Using CUDA libraries for developing efficient GPU applications

# Evaluation

## Graded lab assignments
Each lab assignment should be sent to the address `oguz.kaya[at]universite-paris-saclay.fr` with the subject format **"M2DIPACGPU LABX SURNAME(s) Name(s)"** (e.g., **M2DIPACGPU LAB3 ARNAULT-BARRAT Jean-François**) by 23:59 Saturday following the lab session. Solutions will be posted on the git repository just after this deadline; therefore, do not submit late! Please follow this format for the email subject **EXACTLY TO THE LETTER** as I will employ filters to sort these submissions.

Please only attach source files (\*.cu) to your email, **one .cu file per exercise** (e.g., if the assignment has four exercises, your submission must have four cu files), and please **do not zip the files**!

If there are plots in some exercises, **do not** include them in the submission; those are only for your better understanding of the code's behavior, not for evaluation. If there are other questions/interpretations requiring textual response as part of an exercise, you can put your  at the very beginning of the corresponding source cu file in a comment section.

Out of all lab assignments, two of them will be randomly selected for grading, each providing up to 1/20 points.

If you do not follow the e-mail subject format in at least one of your submissions, you will lose half a point (0.5/20).

## End-of-session mini-exams
At the end of weeks 3, 4, 5, 6 there will be mini-exams of 20-25 minutes, giving 4.5/20 points each. Types of exercises that might appear on these mini-exams are:

**Debugging:** Given a complete code with multiple bugs, the goal is to identify each bug (without correcting them), say what the problem is in a single sentence, and point out the corresponding line(s) in the code.

**Output:** Given a complete code, the goal is to find out what the program could potentially display. Indeed, this requires a thorough understanding of how the code works. There might be multiple possibilities, in which case you should indicate all of them.

**Coding:** A squeleton code or a function signature will be given for a specific task, and you will be expected to provide a parallel/optimized code that performs the intended task.

**Course concepts:** Small questions requiring interpretations from the concepts learned in the class. Example: How come GPUs are able to make very fast context switch between warps in a block?

**Code deciphering:** Given a complete GPU code (with anonymized variable names), understand and explain what the code actually computes.

You can bring **four** A4 sheets to the mini-exam (can use both sides, put whatever you want); no other material is allowed.

# Technical tips

Text editor: If you do not have a "preferred" text editor for development already, I recommend Visual Studio Code: https://code.visualstudio.com . You can follow the official tutorial for the basics: https://www.youtube.com/watch?v=B-s71n0dHUk&list=PLj6YeMhvp2S5UgiQnBfvD7XgOMKs3O_G6

You will need a NVCC compiler for this course. You can use the Godbolt compiler to run your code: cuda.godbolt.org
