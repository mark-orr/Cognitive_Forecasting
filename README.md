# Cognitive_Forecasting

./sandbox/:
    Was used to develop the method in general over a month or so of time March
    and some of april. In this development, the bayes model py functs were just
    the header of each file.  Developed a package later in ../bayes_pack and
    tested in ../bayes_pack_testing

    Has a few key files:
        bayes_test.py provides place to see the workings of the individual
	methods in the bayes model functions and calls the testing version of
	../bayes_pack_testing.

        file5_bayespack_example.py is refactor of sandbox format of files to
	to the py package kind, (e.g., just had to incorporate N into some of the
	function  calls, e.g.).  So, this file is the one to use moving forward
	for publication.

./bayes_pack_testing/:
    This is the package that is a copy of ./bayes_pack with some mods to see
    output of inners of function calls.  Uses ../sandbox/bayes_test.py as the
    main calling function (at least in development to date).

./simulations/:
    This is where we put the final sims for the publication.  See its read me.
    These files should call ./bayes_pack/.




