#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

typedef struct {
  PyObject_HEAD
  double data;
  double grad;
  PyObject* _prev;
  PyObject* _backward;
} Value;

static PyTypeObject Value_Type;

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
    value->_backward = Py_None;
    value->data = data;
    if (_prev) {
      Py_INCREF(_prev);
      value->_prev = _prev;
    } else {
      value->_prev = PyTuple_New(0);
    }
  }
  return (PyObject*)value;
}

static int
Value_traverse(Value *self, visitproc visit, void *arg) {
  int vret;
  if (self->_prev) {
    vret = visit(self->_prev, arg);
    if (vret != 0)
      return vret;
  }
  return 0;
}

static int
Value_clear(Value *self) {
  PyObject *tmp;
  tmp = self->_prev;
  self->_prev = NULL;
  Py_XDECREF(tmp);
  return 0;
}

static void
Value_dealloc(Value* self)
{
  // PyObject_GC_UnTrack(self);
  Value_clear(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Value_str(PyObject *self)
{
  char str[128];
  double data = ((Value*)self)->data;
  double grad = ((Value*)self)->grad;
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

static PyObject *
Value_setgrad(Value* self, PyObject *args) {
  double grad;
  PyArg_ParseTuple(args, "d", &grad);
  self->grad = grad;
  Py_RETURN_NONE;
}

static PyMethodDef Value_methods[] = {
    {"set_grad", (PyCFunction)Value_setgrad, METH_VARARGS, "Set Gradient"},
    {NULL}  /* Sentinel */
};

static PyMemberDef Value_members[] = {
   {"data", T_DOUBLE, offsetof(Value, data), 0, "Data"},
   {"grad", T_DOUBLE, offsetof(Value, grad), 0, "Gradient of the Object"},
   {"_backward", T_OBJECT_EX, offsetof(Value, _backward), 0, "Backward Function"},
   {"_prev", T_OBJECT_EX, offsetof(Value, _prev), 0, "Children"},
   {NULL}
  };

PyObject * pyvalue_add(PyObject * self, PyObject * other) {
  Value* _self = (Value *)self;
  Value* _other = (Value *)other;
  Value* value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  value->data = _self->data + _other->data;
  value->grad = 0.0;
  value->_prev = PyTuple_Pack(2, self, other);
  static char * code = "(value._prev[0].set_grad(value._prev[0].grad + value.grad),\
                         value._prev[1].set_grad(value._prev[1].grad + value.grad))";
  PyCodeObject *c = (PyCodeObject *) Py_CompileString(code, "fn", Py_eval_input);
  c->co_name = PyUnicode_FromString("add_backward");
  c->co_flags |= CO_VARARGS; 
  c->co_nlocals = 1; 
  PyObject * l = PyFunction_New((PyObject *) c, PyEval_GetGlobals());
  value->_backward = l;
  return (PyObject*)value;
}

PyNumberMethods Value_as_number = {
    pyvalue_add,          /* nb_add */
    0, /* nb_subtract */
    0, /* nb_multiply */
    0, /* nb_divide */
    0, /* nb_remainder */
    0, /* nb_divmod */
    0, /* nb_power */
    0, /* nb_negative */
    0, /* nb_positive */
    0, /* nb_absolute */
    0, /* nb_bool */
    0, /* nb_invert */
    0, /* nb_lshift */
    0, /* nb_rshift */
    0, /* nb_and */
    0, /* nb_xor */
    0, /* nb_or */
    0, /* nb_int */
    0, /* nb_float */
    0, /* nb_inplace_add */
    0, /* nb_inplace_subtract */
    0, /* nb_inplace_multiply */
    0, /* nb_inplace_remainder */
    0, /* nb_inplace_power */
    0, /* nb_inplace_lshift */
    0, /* nb_inplace_rshift */
    0, /* nb_inplace_and */
    0, /* nb_inplace_xor */
    0, /* nb_inplace_or */
    0, /* nb_floor_divide */
    0, /* nb_true_divide */
    0, /* nb_inplace_floor_divide */
    0, /* nb_inplace_true_divide */
    0, /* nb_index */
    0, /* nb_matrix_multiply */
    0, /* nb_inplace_matrix_multiply */
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
   &Value_as_number,
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
   (traverseproc)Value_traverse,
   (inquiry)Value_clear,
   0,
   0,
   0,
   0,
   Value_methods,
   Value_members,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   Value_new
  };


static struct PyModuleDef moduledef = {
   PyModuleDef_HEAD_INIT,
   "engine",
   "Gradient Engine",
   -1,
   NULL,
   NULL,
   NULL,
   NULL,
   NULL,
  };

PyMODINIT_FUNC PyInit_engine(void) {
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
