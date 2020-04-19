#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

typedef struct {
  PyObject_HEAD
  double data;
  double grad;
  PyObject* _prev;
  void (*_backward) ();
} Value;

static PyTypeObject Value_Type;

static void
Value_dealloc(Value* self)
{
  // Py_XDECREF(self->_backward);
  // Py_XDECREF(self->_prev);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *
Value_new(PyTypeObject *type,
          PyObject *args,
          PyObject *kwds) {
  double data;
  PyObject* _prev = NULL;
  if (!PyArg_ParseTuple(args, "d|O!", &data, &PyTuple_Type, &_prev)){
    printf("asdasd\n");
    return NULL;
  }
  Value* value = (Value*)Value_Type.tp_alloc(&Value_Type, 0);
  if (value) {
    value->grad = 0.0;
    value->_backward = NULL;
    value->data = data;
    if (_prev) {
      Py_ssize_t n = PyTuple_Size(_prev);
      for (int i = 0; i < n; ++i) {
          printf("%lf\n", ((Value*)PyTuple_GET_ITEM(_prev, i))->data);
          // PyTuple_SetItem(value->_prev, i, PyTuple_GET_ITEM(_prev, i));
      }
      Py_INCREF(_prev);
      value->_prev = _prev;
      printf("%d\n", PyTuple_Check((PyTupleObject *)(value->_prev)));
      printf("%lf\n", ((Value*)PyTuple_GET_ITEM(value->_prev, 0))->data);
      printf("%lf\n", ((Value*)PyTuple_GET_ITEM(value->_prev, 1))->data);
    }
  }
  return (PyObject*)value;
}

static PyObject *
Value_str(PyObject *self)
{
    char str[128];
    double data = ((Value*)self)->data;
    double grad = ((Value*)self)->grad;
    // double test = ((Value*)(((Value*)self)->_prev))->data;
    // sprintf(str, "Value(data=%lf, grad=%lf, %lf)", data, grad, test);
    sprintf(str, "Value(data=%lf, grad=%lf)", data, grad);
    return PyUnicode_FromString(str);
}

static PyObject *
Value_repr(PyObject *self)
{
    char str[128];
    double data = ((Value*)self)->data;
    double grad = ((Value*)self)->grad;
    sprintf(str, "Value(data=%lf, grad=%lf)", data, grad);
    return PyUnicode_FromString(str);
}

static PyMemberDef Value_members[] = {
    {"data", T_DOUBLE, offsetof(Value, data), 0, "Data"},
    {"grad", T_DOUBLE, offsetof(Value, grad), 0, "Gradient of the Object"},
    // {"_backward", T_OBJECT, offsetof(Value, _backward), 0, "Backward Function"},
    {"_prev", T_OBJECT_EX, offsetof(Value, _prev), 0, "Children"},
    {NULL}
};

static PyTupleObject *
Value_getprev(Value *self, void *closure)
{
    Py_INCREF(self->_prev);
    return (PyTupleObject*)(self->_prev);
}

static PyGetSetDef Value_getseters[] = {
    {"_prev", (getter)Value_getprev, NULL, "Children", NULL},
    {NULL} 
};

static PyMethodDef module_functions[] = {
    {NULL, NULL, 0, NULL}
};

static PyTypeObject Value_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Value",
    sizeof(Value),
    0,
    (destructor)Value_dealloc,
    0,
    0,
    0,
    0,
    Value_repr,
    0,
    0,
    0,
    0,
    0,
    Value_str,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Stores a single scalar value and its gradient.",
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Value_members,
    Value_getseters,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Value_new
};


  double data;
  double grad;
  void (*_backward) ();
  PyObject* _prev[2];

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "engine",
    "Gradient Engine",
    -1,
    module_functions,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_engine(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    if (PyType_Ready(&Value_Type) < 0) {
        return NULL;
    }
    Py_INCREF(&Value_Type);
    PyModule_AddObject(m, "Value", (PyObject*)&Value_Type);
    return m;
}
