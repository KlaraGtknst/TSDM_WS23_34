import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

__author__ = "Christian Gruhl"
__copyright__ = "Copyright 2018, Universität Kassel"
__status__ = "Prototype"

cdef extern from "OrthogonalPolynomialSlidingWindow.hpp":
    cdef cppclass OrthogonalPolynomialSlidingWindow:
        OrthogonalPolynomialSlidingWindow(int, int) except +
        void restart()
        void update(double y)
        double eval(double x)
        vector[double] getOrthogonalCoefficients()
        double getApproximationError()

cdef extern from "GrowingWindow.cpp":
    cdef cppclass GrowingWindow:
      GrowingWindow(unsigned int) except +
      void reset()
      void update(double y)
      void eval(double x)
      int getNumSamples()
      vector[double] getOrthogonalCoefficients()
      double getApproximationError()
      int getExtremumPosition()

cdef class OrthogonalPoly:
    """ Orthogonal Polynomial Approximation for univariate time series.
    This is an implementation of an sliding window.
     """
    cdef OrthogonalPolynomialSlidingWindow* swindow
    cdef object _vupdate
    cdef object _veval
    cdef int size

    def __cinit__(self, int degree=3, int size=10):
        """ Initialize a sliding window with the given size and given polynomial degree.
        
        Parameters:
        ===========
        degree : number of polynomial degrees

        size : number of observations used to update the approximation

        """
        self.size = size
        self.swindow = new OrthogonalPolynomialSlidingWindow(degree, size)
        self._vupdate = np.vectorize(self._update)
        self._veval = np.vectorize(self._eval)

    def _update(self, double y):
        """ Update the approximation. If the window is at capacity it will move one step to the right. """
        self.swindow.update(y)

    def _eval(self, double t):
        """ Evaluate the approximation at the given position.
        *NOTE*: since this is a sliding window the 'first' value is at 0
        and the last at 'size' on the x-axis. This might not be intuitive!
        """
        return self.swindow.eval(t)

    def transform(self, np.ndarray[np.double_t, ndim=1] Y):
        """ Update the approximation with the values in Y.
        If len(Y) > size the update will be performed on all instances, but only the last len(Y) will be represented
        by the polynomial.

        Parameters:
        ===========
        Y : y-values to be used for updating

        Returns:
        ========
        The polynomial coefficients (= an observation in Shape-Space)
        """
        self._vupdate(Y)
        return self.coefficients_

    def transform_evaluate(self, np.ndarray[np.double_t, ndim=1] Y):
        """ Update and evaluate the approximation.

        This method updates the polynomial with the given observations Y, evaluates the 
        polynomial at the newly added locations and calculates the approximation error.

        Parameters:
        ===========
        Y : y-values to be used for updating

        Returns:
        ========
        (coefs, y, mse)

        coefs: The polynomial coefficients (= an observation in Shape-Space)
        y : the evaluation of the polynomial at the last len(Y) positions
        mse : the mean squared error for the given observations in Y.
        
        """
        coeffs = self.transform(Y)
        
        x_vals = min(len(Y), self.size)
        X = np.arange(self.size - x_vals, self.size, dtype=np.double)
        
        prediction = self.predict(X)
        mse = np.mean((Y[-len(Y):] - prediction)**2)

        return coeffs, prediction, mse

    def predict(self, np.ndarray[np.double_t, ndim=1] X):
        """ Evaluate the polynomial at the given positions.
        
        *NOTE*: due to its sliding window and time series nature the approximation
        will always begin at 0 and end on **size**!

         """
        return np.asarray(self._veval(X))

    def clear(self):
        """ Clear everything and start with an empty window """
        self.swindow.restart()

    @property
    def coefficients_(self):
        """ The coefficients for the current approximation """
        return self.swindow.getOrthogonalCoefficients()

    @property
    def size_(self):
        return self.size

    def __dealloc__(self):
        del self.swindow

cdef class GrowingPoly:
    cdef GrowingWindow* gwindow
    cdef object _vupdate
    cdef object _veval

    def __cinit__(self, int degree=3):
        self.gwindow = new GrowingWindow(degree)
        self._vupdate = np.vectorize(self._update)
        self._veval = np.vectorize(self._eval)

    cpdef _update(self, double y):
        """ Update the approximation."""
        self.gwindow.update(y)

    cpdef _eval(self, double t):
        """ Evaluate the approximation at the given position."""
        return self.gwindow.eval(t)

    def transform(self, np.ndarray[np.double_t, ndim=1] Y):
        """ Update the approximation with the values in Y.

        Parameters:
        ===========
        Y : y-values to be used for updating

        Returns:
        ========
        The polynomial coefficients (= an observation in Shape-Space)
        """
        self._vupdate(Y)
        return self.coefficients_

    def transform_evaluate(self, np.ndarray[np.double_t, ndim=1] Y):
        """ Update and evaluate the approximation.

        This method updates the polynomia with the given observations Y, evaluates the 
        polynomial at the newly added locations and calculates the approximation error.

        Parameters:
        ===========
        Y : y-values to be used for updating

        Returns:
        ========
        (coefs, y, mse)

        coefs: The polynomial coefficients (= an observation in Shape-Space)
        y : the evaluation of the polynomial at the last len(Y) positions
        mse : the mean squared error for the given observations in Y.
        
        """
        coeffs = self.transform(Y)
        
        x_vals = min(len(Y), self.size)
        X = np.arange(self.size - x_vals, self.size, dtype=np.double)
        
        prediction = self.predict(X)
        mse = np.mean((Y[-len(Y):] - prediction)**2)

    def predict(self, np.ndarray[np.double_t, ndim=1] X):
        """ Evaluate the polynomial at the given positions."""
        return np.asarray(self._veval(X))

    def clear(self):
        """ Clear everything and start with an empty window """
        self.gwindow.reset()

    @property
    def coefficients_(self):
        """ The coefficients for the current approximation """
        return self.gwindow.getOrthogonalCoefficients()

    @property
    def size_(self):
        return self.gwindow.getNumSamples()

    def __dealloc__(self):
        del self.gwindow

cdef class OrthogonalPolyDB(OrthogonalPoly):
    """ Double Buffer implementation to circumvent bug in C++ Implementation.
    After a given number of updates a new OrthogonalPolynomialSlidingWindow is used for
    approximation.
    When the counter reaches the window size the double buffer is fitted in parallel 
    for a seamless switch between both windows.
    """
    
    cdef int reset_after
    cdef int reset_counter
    cdef int fill_db
    cdef degree
    cdef OrthogonalPolynomialSlidingWindow* buffer

    def __cinit__(self, int degree=3, int size=10):
        """ Initialize a sliding window with the given size and given polynomial degree.
        
        Parameters:
        ===========
        degree : number of polynomial degrees

        size : number of observations used to update the approximation

        reset_after : number of update steps (samples) after which the double budder is blitted.

        """
        # super().__cinit__ is automatically called
        self.degree = degree
        self.reset_after = 500
        self.reset_counter = self.reset_after
        self.buffer = new OrthogonalPolynomialSlidingWindow(self.degree, self.size)
    
    def set_reset_after(self, int reset_after):
        """ Sets the reset_after attribute.

            This attribute cannot be added to the constructor, because in cython
            the parent constructor is automatically called with the same set of 
            parameters.
        """
        self.reset_after = reset_after
        self.reset_counter = self.reset_after

    cpdef _update(self, double y):
        self.reset_counter -= 1
        if self.reset_counter <= self.size:
            if self.reset_counter == 0:
                del self.swindow
                self.swindow = self.buffer
                self.buffer = new OrthogonalPolynomialSlidingWindow(self.degree, self.size)
                self.reset_counter = self.reset_after
            else:
                self.buffer.update(y)
        self.swindow.update(y)

    def clear(self):
        """ Clear everything and start with an empty window """
        del self.buffer
        self.buffer = new OrthogonalPolynomialSlidingWindow(self.degree, self.size)        
        self.reset_counter = self.reset_after
        super().clear()
    
    def __dealloc__(self):
        del self.buffer
