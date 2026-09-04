# Chapter 1 --- Overview of Operating System

> **Source:** Chapter 1 notes PDF\
> **Format:** Clean Markdown notes with the diagrams recreated in
> Mermaid so they remain editable/renderable in Markdown editors that
> support Mermaid.

------------------------------------------------------------------------

## 1.1 OS --- Concept, Components of OS, Operations of OS, Views of OS

### Operating System --- Concept

A computer system has many resources (hardware and software) required to
complete a task.

Commonly required resources include:

-   Input/Output devices
-   Memory
-   File storage space
-   CPU
-   Processor
-   Files

**Operating System (OS)** is a program that manages the computer
hardware. It provides a basis for application programs and acts as an
**intermediary between the computer user and the computer hardware**.

An operating system performs tasks such as:

-   File management
-   Memory management
-   Process management
-   Handling input and output
-   Controlling peripheral devices such as disk drives and printers

**Examples of Operating Systems:** Linux, Windows, OS X, VMS, OS/400,
AIX, z/OS, etc.

### Definition

> **Operating System is an interface between the computer user and
> computer hardware.**

OR

> **An operating system is a program that acts as an interface between
> the user and computer hardware and controls the execution of all kinds
> of programs.**

### Diagram --- Operating System

``` mermaid
flowchart TB
    U1[User 1] --> OS[Operating System]
    U2[User 2] --> OS
    Un[User n] --> OS
    OS --> SS[System Software]
    OS --> AS[Application Software]
    OS --> H[Hardware]
    H --> CPU[CPU]
    H --> RAM[RAM]
    H --> IO[I/O]
```

------------------------------------------------------------------------

## OS as a Control Program

OS acts as a **control program** which controls all activities of the
computer system.

It interacts with computer hardware to perform different tasks.

### Primary Objective of OS

The primary objective of an OS is:

1.  To make the computer convenient to use.
2.  To utilize hardware in an efficient manner.

------------------------------------------------------------------------

# Components of a Computer System

A computer system can be divided into **4 components**:

1.  Hardware
2.  Operating System
3.  Application Programs
4.  Users

### Diagram --- Components of Computer System

``` mermaid
flowchart TB
    U[Users]
    APP[System and Application Programs]
    OS[Operating System]
    HW[Computer Hardware]

    U --> APP
    APP --> OS
    OS --> HW
```

### 1. Hardware

Hardware consists of the physical parts of a machine which provide basic
computing resources.

Examples:

-   CPU
-   Memory
-   I/O devices

These provide basic computing resources for the system.

### 2. Application Program

Application programs define the way in which these resources are used to
solve user problems.

Examples:

-   Word processor
-   Compiler
-   Browser
-   Text editor
-   Database system

### 3. Operating System

The operating system controls and coordinates the use of hardware among
the various application programs for various users.

### 4. Users

Users may be:

-   Human beings
-   Machines
-   Other computers

------------------------------------------------------------------------

# Operating System as an Environment

An operating system is similar to a **government**.

Like a government, it does not perform a useful function by itself. It
simply provides an environment in which other programs can do useful
work.

To understand the operating system's role, it can be viewed from two
viewpoints:

1.  User View
2.  System View

------------------------------------------------------------------------

# Views of OS

## 1. User View

The user's view of the computer varies according to the interface being
used.

### PC / Single-user system

Most computer users sit in front of a PC consisting of:

-   Monitor
-   Keyboard
-   Mouse
-   System unit

The goal is to maximize the work or play that the user is performing.

In this case, the OS is designed mostly for **ease of use**, with some
attention paid to resource utilization.

Performance is important to the user, and such systems are generally
optimized for the **single-user experience** rather than the
requirements of multiple users.

### Mainframe / Minicomputer

A user may sit at a terminal connected to a mainframe or minicomputer
while other users access the same computer through other terminals.

Users share resources and may exchange information.

The OS is designed to:

-   Maximize resource utilization
-   Use available CPU time efficiently
-   Use memory efficiently
-   Use I/O efficiently
-   Ensure that no individual user takes more than a fair share

### Workstations connected through networks

Users may sit at workstations connected to networks of other
workstations.

They have dedicated resources available at their workstation but may
also share:

-   Networking resources
-   Servers
-   Compute servers
-   Print servers

The OS therefore needs to balance:

-   Individual usability
-   Resource utilization

### Handheld computers

Many varieties of handheld computers have become common.

Most are standalone units for individual users, while some are connected
to networks directly or through wireless modems and networking.

Because of hardware and interface limitations, they perform relatively
few functions.

Their OS is designed mainly for:

-   Individual usability
-   Performance
-   Battery-life efficiency

### Computers with little or no user view

Some computers have little or no user view.

------------------------------------------------------------------------

# 2. System View

From the computer's point of view, the operating system is the program
most intimately involved with the hardware.

The OS can be viewed as a **resource allocator**.

A computer system has many resources required to solve a problem, such
as:

-   CPU time
-   Memory space
-   Storage space
-   I/O devices

The operating system acts as the **manager of these resources**.

Resource management is especially important when many users access a
mainframe or minicomputer.

### OS as a Control Program

Another view emphasizes the need to control various devices and user
programs.

An operating system is a **control program** that:

-   Manages execution of user programs
-   Prevents errors
-   Ensures proper use of the computer
-   Controls I/O devices

------------------------------------------------------------------------

# Operations of OS

The operating system performs / manages the following operations:

1.  Transferring input from an input device to memory.
2.  Transferring input from memory to CPU.
3.  Transferring data from CPU to an output device.
4.  Copying or moving data from one storage device to another.
5.  Managing programs running on the system.
6.  Displaying appropriate error messages in case of software or
    hardware problems.
7.  Maintaining details of files and directories.
8.  Providing security to user data.
9.  Protecting storage devices from overwriting.
10. Managing hardware and software resources of the system.
11. Providing a stable and consistent way for applications to deal with
    hardware without knowing all the hardware details.

------------------------------------------------------------------------

# OS Management Functions

The notes identify the following major management areas:

-   Program Management
-   Resource Management
    -   Memory Management
    -   File System Management
    -   I/O Subsystem Management
-   Security and Protection
-   Process Management

------------------------------------------------------------------------

## Process Management

A **process is a program in execution**.

It is a unit of work within the system.

### Program vs Process

-   **Program:** Passive entity
-   **Process:** Active entity

A process needs resources to accomplish its task:

-   CPU
-   Memory
-   I/O
-   Files
-   Initialization data

When a process terminates, reusable resources must be reclaimed.

### Single-threaded Process

A single-threaded process has one program counter specifying the
location of the next instruction to execute.

The process executes instructions sequentially until completion.

### Multi-threaded Process

A multi-threaded process has one program counter per thread.

Typically, a system has many processes, some of which are
operating-system processes, running concurrently on one or more CPUs.

### Concurrency

Concurrency is achieved by **multiplexing the CPUs among processes /
threads**.

### Process Management Activities

The OS is responsible for:

-   Creating and deleting both user and system processes
-   Suspending and resuming processes
-   Providing mechanisms for process synchronization
-   Providing mechanisms for process communication
-   Providing mechanisms for deadlock handling

------------------------------------------------------------------------

# Memory Management

To execute a program, all or part of the instructions must be in memory.

All or part of the data needed by the program must also be in memory.

**Memory management determines what is in memory and when.**

Main goals include:

-   Optimizing CPU utilization
-   Improving computer response to users

### Memory Management Activities

-   Keeping track of which parts of memory are currently being used and
    by whom
-   Deciding which processes (or parts of processes) and data should be
    moved into and out of memory
-   Allocating memory space as needed
-   Deallocating memory space as needed

------------------------------------------------------------------------

# File-System Management

The OS provides a **uniform, logical view of information storage**.

It abstracts physical properties into a logical storage unit called a
**file**.

### File-System Management

-   Files are usually organized into directories.
-   Access control is provided on most systems to determine who can
    access what.

### OS File-System Activities

-   Creating files
-   Deleting files
-   Creating directories
-   Deleting directories
-   Providing primitives to manipulate files and directories
-   Mapping files onto secondary storage
-   Backing up files onto stable (non-volatile) storage media

------------------------------------------------------------------------

# I/O Subsystem

One purpose of the OS is to **hide the peculiarities of hardware devices
from the user**.

The I/O subsystem is responsible for:

### 1. Buffering

Storing data temporarily while it is being transferred.

### 2. Caching

Storing parts of data in faster storage for better performance.

### 3. Spooling

Overlapping the output of one job with the input of another job.

### Other responsibilities

-   Providing a general device-driver interface
-   Providing drivers for specific hardware devices

------------------------------------------------------------------------

# Protection and Security

### Protection

**Protection** is any mechanism for controlling access of processes or
users to resources defined by the OS.

### Security

**Security** is the defense of the system against internal and external
attacks.

Examples of threats include:

-   Denial-of-service
-   Worms
-   Viruses
-   Identity theft
-   Theft of service

Systems generally distinguish among users to determine **who can do
what**.

### User Identity

User identities (**user IDs / security IDs**) identify users and are
associated with a number.

The user ID is associated with:

-   Files
-   Processes
-   Access control

### Group Identifier

A **group identifier (group ID)** allows a set of users to be defined
and controls to be managed for the group.

The group ID can be associated with:

-   Processes
-   Files

### Privilege Escalation

**Privilege escalation** allows a user to change to an effective ID with
more rights.

------------------------------------------------------------------------

# 1.2 Types of Operating System

The notes cover:

1.  Batch Operating System
2.  Multiprogramming
3.  Multitasking
4.  Time-Sharing OS
5.  Multiprocessor OS / Tightly Coupled / Parallel System
6.  Distributed Operating System / Loosely Coupled System
7.  Real-Time System
8.  Mobile OS

------------------------------------------------------------------------

# 1. Batch Operating System

A **Batch OS** is used for a sequence of user jobs.

Users leave their programs with the operator. The operator sorts
programs with similar requirements into batches and submits them to the
**Batch Processing System (BPS)**.

The jobs are then executed automatically without user interaction.

### Important Points

-   Users do not interact with the computer directly.
-   Jobs are recognized using special markers indicating the start and
    end of jobs.
-   BPS consists of a **batch monitor / supervisor** as a major
    component.
-   The monitor is permanently placed in part of memory.
-   The remaining memory is used to process user jobs.

### Batch Processing Diagram

``` mermaid
flowchart LR
    U1[User 1<br/>Job 1] --> O[Operator]
    U2[User 2<br/>Job 2] --> O
    Un[User n<br/>Job n] --> O
    O --> B[Batch 1 / Batch 2 / ...]
    B --> BM[Batch Monitor]
    BM --> CPU[CPU]
    CPU --> OUT[Output]
```

## Functions of Batch Monitor

The batch monitor keeps control of the processing environment.

### a. Scheduling

Scheduling determines which service request should be handled next.

The notes mention the **First Come First Serve (FCFS)** algorithm for
scheduling.

### b. Memory Management

During BPS operation, memory is divided into:

1.  System working area
2.  User working area

Partitioning of memory is done by the OS.

Some part of the monitor is permanently stored in memory. This is known
as the **resident area of monitor**.

### c. Sharing and Protection

By dividing memory into two parts, protection is provided.

------------------------------------------------------------------------

## Advantages of Batch OS

-   Repeated jobs are done fast without user interaction.
-   Special hardware and system support are not required to input data
    in batch systems.
-   Best for large organizations.
-   Batch systems can be shared by multiple users.
-   Idle time is very less.
-   Specific times can be assigned to batch jobs, such as when the
    computer is idle or at night.
-   Large repeated work can be managed easily.
-   Programmer attention is reduced.

## Disadvantages of Batch OS

-   Monitor remains in memory, resulting in memory wastage.
-   Difficult to debug batch systems.
-   Batch systems can be costly.
-   Lack of protection.

------------------------------------------------------------------------

# 2. Multiprogramming

The process of executing **multiple programs simultaneously** is known
as multiprogramming.

### Why Multiprogramming?

In uniprogramming:

-   CPU cannot be kept busy all the time.
-   I/O devices cannot be kept busy all the time.

Using multiprogramming, utilization of CPU and I/O devices can be
increased.

A multiprogramming OS keeps several jobs in memory at the same time.

The set of jobs is kept in a **job pool**.

### Memory Layout of Multiprogramming

``` mermaid
flowchart TB
    OS[Operating System]
    J1[Job 1]
    J2[Job 2]
    J3[...]
    JN[Job n]
    E[Empty Space]

    OS --> J1
    J1 --> J2
    J2 --> J3
    J3 --> JN
    JN --> E
```

### Working of Multiprogramming

Suppose Job A is executing.

1.  Job A may require an I/O operation.
2.  Instead of keeping the CPU idle, the OS switches to another job.
3.  The next job executes until it requires I/O or another wait.
4.  The CPU then switches to another available job.
5.  This continues as required.

Thus, CPU time is utilized while another job is waiting for I/O.

**Example:** Running Excel and Firefox at the same time.

### Advantages

-   Time is reduced.
-   Efficiency is improved.
-   Waiting time is limited for programs.
-   CPU utilization increases.

### Disadvantages

-   Proper memory management is required.
-   Proper scheduling algorithm is required.
-   Maintenance is expensive.

------------------------------------------------------------------------

# 3. Multitasking

**Multitasking** is the ability of an operating system to execute more
than one task on a single-processor machine.

It is a logical extension of multiprogramming.

### Important Point

On a single processor, two tasks are not actually executed at exactly
the same time.

The CPU switches from one task to another so quickly that it appears as
if all tasks are executing simultaneously.

### Task

A task is a particular operation such as:

-   Edit
-   Copy
-   Rename
-   Delete
-   Open
-   Close

### Foreground and Background Tasks

Multitasking can be achieved by:

-   Executing one job in the **foreground**
-   Executing many jobs in the **background**

**Foreground jobs** require user interaction.

**Background jobs** do not require user interaction.

**Example:** While editing a document, a user can take printouts and
listen to music.

### Multitasking Diagram

``` mermaid
flowchart TB
    B[Browser]
    E[Excel]
    P[Paint]
    M[Music]
    OS[Operating System]
    CPU[CPU]

    B --> OS
    E --> OS
    P --> OS
    M --> OS
    OS --> CPU
```

### Advantages

-   Multiple tasks can be run seemingly simultaneously.
-   Speed of execution increases.
-   CPU utilization increases.

### Disadvantages

-   Systems may not be reliable.
-   Data communication problems may occur.
-   It raises security concerns for user programs and data.

## Multiprogramming vs Multitasking

  -----------------------------------------------------------------------
  Sr. No.                 Multiprogramming        Multitasking
  ----------------------- ----------------------- -----------------------
  1                       Multiple programs are   Multiple tasks are
                          executed                executed seemingly
                          simultaneously.         simultaneously.

  2                       Collection of different A task is a smaller
                          tasks is a program.     unit of a program.

  3                       A program may contain   A task does not contain
                          multiple tasks.         a complete program.

  4                       User interaction is not User interaction is
                          provided.               provided.

  5                       Utilizes CPU and I/O    Utilizes CPU
                          devices efficiently.    efficiently.

  6                       Focus is on keeping     Focus is on handling
                          multiple programs       multiple tasks
                          available for           interactively.
                          execution.              
  -----------------------------------------------------------------------

There are two types of tasks:

-   Foreground task
-   Background task

------------------------------------------------------------------------

# 4. Time-Sharing OS

**Time-sharing** is a logical extension of multiprogramming in which
processor time is shared among multiple users simultaneously.

### Main Objective

The main objective of time-sharing is to **minimize response time**.

Multiple jobs are executed by the CPU by switching between them.

The switches occur very frequently, allowing the user to receive an
immediate response.

**Example OS:** Linux

### Time-Sharing Diagram

``` mermaid
flowchart TB
    C[Large Central System]
    T1[Terminal 1]
    T2[Terminal 2]
    T3[Terminal 3]
    TN[Terminal n]

    C --> T1
    C --> T2
    C --> T3
    C --> TN
```

A large central computer provides a **time slice** (in milliseconds) at
regular intervals to each terminal.

The computer switches between terminals so rapidly that each terminal
feels that the computer is dedicated to that user.

A job does not have to wait for every operation; when the time slice is
very small, the CPU quickly moves to another job.

### Advantages

-   Quick response
-   Avoids duplication of software
-   Reduces CPU idle time

### Disadvantages

-   Expensive to build
-   Data communication can be complicated

------------------------------------------------------------------------

# 5. Multiprocessor OS / Tightly Coupled / Parallel System

In a multiprocessor system, **more than one processor** can be placed in
a single cabinet.

Processors share:

-   Resources
-   Bus
-   Memory

### Objective

To get the **maximum work done in less time**.

It is also called:

-   Tightly coupled system
-   Parallel system

### Multiprocessor Diagram

``` mermaid
flowchart TB
    R[Shared Memory / BUS / Resources]
    C1[CPU 1]
    C2[CPU 2]
    CN[CPU n]

    R --- C1
    R --- C2
    R --- CN
```

### Multiprocessing System

Applications such as:

-   Word processor
-   E-mail
-   Web browser
-   Antivirus

can use processor cores through the operating system.

``` mermaid
flowchart TB
    A1[Word Processor]
    A2[E-mail]
    A3[Web Browser]
    A4[Antivirus]
    OS[Operating System]
    C1[CPU Core 1]
    C2[CPU Core 2]

    A1 --> OS
    A2 --> OS
    A3 --> OS
    A4 --> OS
    OS --> C1
    OS --> C2
```

## Types of Multiprocessor System

### 1. Symmetric Multiprocessing

In this system:

-   Two or more processors are connected with a high-bandwidth link.
-   They are managed by a single OS.
-   All processors equally share I/O devices.

### 2. Asymmetric Multiprocessing

Each processor is assigned a specific task.

This uses the **Master-Slave concept**:

-   Master processor controls slave processors.
-   Slave processors perform assigned tasks.

## Advantages

-   Increases throughput.
-   Increases reliability.
-   If one processor fails, the whole system does not necessarily fail.
-   More than one processor can be used.
-   Maximum work can be done in less time.

## Disadvantages

-   Expensive to build and maintain.
-   More complex hardware and software compared with a uniprocessor
    system.

------------------------------------------------------------------------

# 6. Distributed Operating System / Loosely Coupled System

A distributed operating system is a **collection of processors
interconnected by a communication network**.

It is also called a **loosely coupled system** and is opposite to a
multiprocessor system.

### Important Points

-   From the point of view of a specific processor, the other processors
    and their resources are remote, while its own resources are local.
-   Data-processing jobs are distributed among processors accordingly.
-   Processors communicate through various communication lines, such as
    high-speed buses or telephone lines.
-   Processors in a distributed system may vary in size and function.
-   Processors are referred to as **sites, nodes, computers**, etc.

### Distributed System Diagram

``` mermaid
flowchart LR
    A[Site A<br/>Server] --- N[Communication Network]
    B[Site B<br/>Client] --- N
    C[Site C<br/>Resources] --- N
```

## Advantages

-   **Resource sharing:** A user at one site may be able to use
    resources available at another site.
-   **Data exchange:** Users can exchange data through electronic mail.
-   If one site fails, the remaining sites can potentially continue
    operating.
-   Better service can be provided to customers.
-   Reduction of load on the host computer.
-   Reduction of delays in data processing.

## Disadvantages

-   Complex and difficult to build.
-   Strong security measures are needed to protect shared resources.

------------------------------------------------------------------------

# 7. Real-Time System

Real-time systems are used in environments where a large number of
events, mostly external, must be accepted and processed by the computer
system in a short time or within a deadline.

They are often used as **control systems**.

### Objective

-   Provide quick event response time.
-   Meet the scheduled deadline.

A real-time system is used when **rigid time requirements** have been
placed on the operation of a process.

### Working

Sensors bring data to the computer.

The computer analyzes the data and adjusts controls according to the
sensor input.

It has fixed time constraints; otherwise, the system may fail.

### Examples

-   Power plant control
-   Industrial control
-   Railway control
-   Airline reservation system
-   Satellite communication
-   Traffic light control

### Real-Time System Diagram

``` mermaid
flowchart LR
    S[Sensors] --> CPU[CPU]
    CPU --> C[Control System]
    C --> O[Controlled Equipment / Process]
    O --> S
```

## Types of Real-Time Systems

### 1. Hard Real-Time System

A hard real-time system **guarantees that critical tasks complete on a
given time constraint**.

A delay in a critical signal may cause system failure.

**Example:** Car engine control system.

### 2. Soft Real-Time System

A soft real-time system can **tolerate some delay in time**.

**Examples:**

-   Live audio
-   Live video

## Advantages

-   Direct communication between user and system is possible.
-   System-generated tasks can be completed within the required time.

## Disadvantages

-   More complex and difficult to maintain.

------------------------------------------------------------------------

# 8. Mobile OS

A **mobile OS** is an operating system that controls:

-   Smartphones
-   PDAs
-   Tablet PCs
-   Other mobile devices

Unlike Linux or Windows operating systems commonly used on
desktop/laptop computers, a mobile operating system is designed for
mobile devices.

A mobile OS is the **software platform on top of which other programs
can run on mobile devices**.

The OS determines the functions and features available on the device,
such as:

-   Thumb wheel
-   Keyboards
-   WAP
-   Synchronization with applications
-   E-mail
-   Text messaging

### Different Mobile OS

The notes list:

-   Android OS --- Google Inc.
-   Bada --- Samsung Electronics
-   BlackBerry OS
-   iPhone OS / iOS --- Apple
-   MeeGo OS --- Nokia and Intel
-   Symbian OS --- Nokia
-   Windows Mobile --- Windows Phone

------------------------------------------------------------------------

# 8.1 Android OS

Android is a **Linux-based operating system** designed primarily for
touchscreen mobile devices such as smartphones and tablet computers.

The notes describe Android as one of the most widely used mobile
operating systems.

It is a powerful operating system and supports a large number of
applications.

Android software is supported by the **ARM architecture platform**.

Android is described in the notes as an **open-source operating
system**, meaning it is free and can be used by anyone.

Android has millions of applications that can help users manage
different activities.

## Advantages of Android

-   Linux-based open-source operating system
-   Can be developed by anyone
-   Easy access to Android apps
-   Battery can be replaced
-   Mass storage / disk drive can be replaced
-   Supports Google services
-   Supports multitasking
-   Free to customize
-   Supports 2D and 3D graphics

## Disadvantages of Android

-   Usually requires more code in Java than Objective-C.
-   Complex layouts and animations are harder to code in Android.
-   Applications containing viruses may also be present in the Android
    Market.
-   Many processes may run in the background, causing the battery to
    drain quickly.
-   Lower security; applications can be installed from unknown resources
    and may steal information.

------------------------------------------------------------------------

# 8.2 iOS

iOS is developed by **Apple**.

It powers Apple's mobile devices, including:

-   iPhone
-   iPad
-   iPod

The notes describe it as the second most popular mobile OS after
Android.

The iOS user interface is based on **direct manipulation using
multi-touch gestures**.

Interface control elements include:

-   Sliders
-   Switches
-   Buttons

Gestures include:

-   Swipe
-   Tap
-   Pinch
-   Reverse pinch

These gestures have specific meanings within the iOS operating system
and its multi-touch interface.

## Advantages of iOS

-   High customer service
-   More security
-   Supports multitasking
-   Good performance
-   Supports a vast number of applications

## Disadvantages of iOS

-   Not flexible; only supports iOS devices.
-   Not open source.
-   Costly application development.
-   Not customizable.

------------------------------------------------------------------------

# Android vs iOS

  -----------------------------------------------------------------------
  Parameter               Android                 iOS
  ----------------------- ----------------------- -----------------------
  Source model            Open source             Closed, with
                                                  open-source components

  Platform                Linux                   OS X, UNIX

  Language Used           C, C++, Java            C++, Objective-C

  Developer               Google, Open Handset    Apple Inc.

  Widgets                 Yes, except on lock     No, except in
                          screen                  Notification Center

  File transfer           Easier than iOS         More difficult

  Internet browsing       Google Chrome           Safari

  Security                Low                     High
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Quick Revision --- Types of OS

  -----------------------------------------------------------------------
  OS Type                             Main Idea
  ----------------------------------- -----------------------------------
  **Batch OS**                        Jobs are collected into batches and
                                      executed without direct user
                                      interaction.

  **Multiprogramming**                Multiple programs are kept in
                                      memory so CPU/I/O utilization can
                                      be improved.

  **Multitasking**                    CPU switches rapidly between
                                      multiple tasks, giving the
                                      appearance of simultaneous
                                      execution.

  **Time-Sharing**                    CPU time is shared among multiple
                                      users using small time slices.

  **Multiprocessor OS**               Multiple processors share resources
                                      in a tightly coupled system.

  **Distributed OS**                  Processors at different sites
                                      communicate through a network.

  **Real-Time OS**                    Tasks must respond and complete
                                      within specified time constraints.

  **Mobile OS**                       OS designed to control smartphones,
                                      tablets and other mobile devices.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Key Definitions for Exam

### Operating System

> An operating system is a program that acts as an interface between the
> user and computer hardware and controls the execution of programs.

### Process

> A process is a program in execution.

### Protection

> Protection is any mechanism for controlling access of processes or
> users to resources defined by the OS.

### Security

> Security is the defense of the system against internal and external
> attacks.

### Batch OS

> A Batch OS processes a sequence of user jobs automatically without
> direct user interaction.

### Multiprogramming

> Multiprogramming is the process of executing multiple programs
> simultaneously by keeping several jobs in memory.

### Multitasking

> Multitasking is the ability of an operating system to execute more
> than one task on a single-processor machine.

### Time-Sharing

> Time-sharing is a system in which processor time is shared among
> multiple users by rapidly switching between jobs.

### Multiprocessor OS

> A multiprocessor OS uses more than one processor in a tightly coupled
> system where processors share resources.

### Distributed OS

> A distributed OS is a collection of processors interconnected by a
> communication network.

### Real-Time System

> A real-time system is a system in which tasks must be processed within
> specified time constraints or deadlines.

### Mobile OS

> A mobile OS is an operating system designed to control mobile devices
> such as smartphones, PDAs and tablets.

------------------------------------------------------------------------

## Source Page Coverage

The uploaded notes contain **21 PDF pages**. Pages that contain only the
repeated "NOTES BY ANSH" watermark / no substantive notes are not
reproduced as separate content sections. The substantive material from
the source is organized above in the same chapter sequence:

-   Chapter 1 / OS concept
-   Components of computer system
-   Views of OS
-   Operations of OS
-   OS management functions
-   Process, memory, file-system, I/O, protection and security
    management
-   Types of OS
-   Batch OS
-   Multiprogramming
-   Multitasking
-   Time-sharing
-   Multiprocessor OS
-   Distributed OS
-   Real-time system
-   Mobile OS
-   Android
-   iOS
-   Android vs iOS comparison

> **Note:** The diagrams in this Markdown file are recreated from the
> diagrams in the PDF using Mermaid rather than pasted as screenshots,
> so they can be edited and rendered in Markdown applications that
> support Mermaid.
