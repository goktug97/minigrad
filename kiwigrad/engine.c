#include <Python.h>
#include "structmember.h"
#include <math.h>
#include <string.h>

typedef struct List List;

typedef struct {
  PyObject_HEAD
  double data;
  double grad;
  double oa;
  PyObject *prev;
  PyObject *op;
  int func_idx;
  PyObject *tmp;
  List *topology;
  int visited;
} Value;

typedef struct _Node {
  struct _Node *prev;
  Value *value;
  struct _Node *next;
} Node;

struct List {
  Node *head;
  Node *tail;
};

static PyTypeObject Value_Type;

PyObject *
Value_new(PyTypeObject *type,
          PyObject *args,
          PyObject *kwds) {
  double data;
  if (!PyArg_ParseTuple(args, "d", &data))
    return NULL;
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  if (value) {
    value->grad = 0.0;
    value->visited = 0;
    value->data = data;
    value->oa = 0.0;
    value->prev = PyTuple_New(0);
    value->op = Py_None;
    value->topology = NULL;
    value->tmp = Py_None;
    value->func_idx = -1;
  }
  return (PyObject *)value;
}

static int
Value_traverse(Value *self, visitproc visit, void *arg) {
  int vret;
  if (self->prev) {
    vret = visit(self->prev, arg);
    if (vret != 0)
      return vret;
  }
  return 0;
}

static int
Value_clear(Value *self) {
  PyObject *tmp;
  tmp = self->prev;
  self->prev = NULL;
  Py_XDECREF(tmp);

  tmp = self->tmp;
  self->tmp = NULL;
  Py_XDECREF(tmp);

  return 0;
}

static void
Value_dealloc(Value *self) {
  // PyObject_GC_UnTrack(self);
  Value_clear(self);
  if (((Value *)self)->topology) {
    Node *node = ((Value *)self)->topology->tail;
    Node *tmp;
    while (node) {
      tmp = node;
      node = node->prev;
      free(tmp);
    }
    free(((Value *)self)->topology);
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Value_str(PyObject *self) {
  char str[128];
  double data = ((Value *)self)->data;
  double grad = ((Value *)self)->grad;
  sprintf(str, "Value(data=%lf, grad=%lf)", data, grad);
  return PyUnicode_FromString(str);
}

static PyObject *
Value_repr(PyObject *self) {
  char str[128];
  double data = ((Value *)self)->data;
  double grad = ((Value *)self)->grad;
  sprintf(str, "Value(data=%lf, grad=%lf)", data, grad);
  return PyUnicode_FromString(str);
}

static PyObject *
relu_backward(PyObject *self) {
  Value *child = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  child->grad += (((Value *)self)->data > 0) * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
sigmoid_backward(PyObject *self) {
  Value *child = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  double x = ((Value *)self)->data;
  child->grad += (x * (1 - x)) * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
log_backward(PyObject *self) {
  Value *child = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  double x = ((Value *)self)->data;
  double t = exp(x);
  child->grad += (pow(t, -1)) * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
tanh_backward(PyObject *self) {
  Value *child = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  double x = ((Value *)self)->data;
  child->grad += (1 - pow(x, 2)) * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
exp_backward(PyObject *self) {
  Value *child = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  double x = ((Value *)self)->data;
  child->grad += x * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
add_backward(PyObject *self) {
  Value *child_1 = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  Value *child_2 = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 1));
  
  child_1->grad += ((Value *)self)->grad;
  child_2->grad += ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
mul_backward(PyObject *self) {
  Value *child_1 = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 0));
  Value *child_2 = ((Value *)PyTuple_GetItem(((Value *)self)->prev, 1));
  child_1->grad += child_2->data * ((Value *)self)->grad;
  child_2->grad += child_1->data * ((Value *)self)->grad;
  Py_RETURN_NONE;
}

static PyObject *
pow_backward(PyObject *self) {
  Value *child = ((Value*)PyTuple_GetItem(((Value *)self)->prev, 0));
  double exponent = ((Value *)self)->oa; 
  double base = child->data;
  child->grad += exponent * pow(base, exponent - 1.0) * ((Value *)self)->grad;
  Py_RETURN_NONE;
}


typedef PyObject * (*BackwardFunction)(PyObject *);
static BackwardFunction backward_methods[] = {
   &add_backward,
   &mul_backward,
   &pow_backward,
   &relu_backward,
   &sigmoid_backward,
   &log_backward,
   &tanh_backward,
   &exp_backward
  };

static PyObject * Value_relu(PyObject *self) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  if (((Value *)self)->data < 0.0) {
    value->data = 0.0;
  } else {
    value->data = ((Value *)self)->data;
  }
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->op = PyUnicode_FromString("relu");
  value->func_idx = 3;
  return (PyObject *)value;
}

static PyObject * Value_sigmoid(PyObject *self) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  double x = ((Value *)self)->data;
  double t = 1 / (1 + exp(-x));
  value->data = t;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->op = PyUnicode_FromString("sigmoid");
  value->func_idx = 4;
  return (PyObject *)value;
}

static PyObject * Value_log(PyObject *self) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  double x = ((Value *)self)->data;
  double t = log(x);
  value->data = t;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->op = PyUnicode_FromString("log");
  value->func_idx = 5;
  return (PyObject *)value;
}

static PyObject * Value_tanh(PyObject *self) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  double x = ((Value *)self)->data;
  double t = (exp(2*x) - 1)/(exp(2*x) + 1);
  value->data = t;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->op = PyUnicode_FromString("tanh");
  value->func_idx = 6;
  return (PyObject *)value;
}

static PyObject * Value_exp(PyObject *self) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  double x = ((Value *)self)->data;
  double t = exp(x);
  value->data = t;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->op = PyUnicode_FromString("exp");
  value->func_idx = 7;
  return (PyObject *)value;
}

static void list_append(List *list, PyObject *value) {
  Node *node = malloc(sizeof(Node));
  node->value = ((Value *)value);
  node->next = NULL;
  if (!(list->head)) {
    node->prev = NULL;
    list->head = node;
    list->tail = node;
  } else {
    node->prev = list->tail;
    list->tail->next = node;
    list->tail = node;
  }
}

static void build_topology(PyObject *value, List *topology) {
  Value *_value = ((Value *)value);
  if (!(_value->visited)) {
    _value->visited = 1;
    int n_child = PyTuple_Size(_value->prev);
    for (int i = 0; i < n_child; ++i) {
      PyObject *child = PyTuple_GetItem(((Value *)value)->prev, i);
      build_topology(child, topology);
    }
    list_append(topology, value);
  }
}

static PyObject *
_backward(PyObject *self) {
  if (((Value *)self)->func_idx >= 0) {
    return backward_methods[((Value *)self)->func_idx](self);
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject *
backward(PyObject *self){
  if (!((Value *)self)->topology) {
    ((Value *)self)->topology = malloc(sizeof(List));
    ((Value *)self)->topology->head = NULL;
    ((Value *)self)->topology->tail = NULL;
    build_topology(self, ((Value *)self)->topology);
  } 
  ((Value *)self)->grad = 1.0;
  Node *node = ((Value *)self)->topology->tail;
  while (node) {
    _backward((PyObject *)(node->value));
    node = node->prev;
  }
  Py_RETURN_NONE;
}

static PyMethodDef Value_methods[] = {
    {"relu", (PyCFunction)Value_relu, METH_NOARGS, "ReLU"},
    {"sigmoid", (PyCFunction)Value_sigmoid, METH_NOARGS, "Sigmoid"},
    {"log", (PyCFunction)Value_log, METH_NOARGS, "Log"},
    {"tanh", (PyCFunction)Value_tanh, METH_NOARGS, "Tanh"},
    {"exp", (PyCFunction)Value_exp, METH_NOARGS, "Exp"},
    // {"_backward", (PyCFunction)_backward, METH_NOARGS, "Backward"}, // DEBUG
    {"backward", (PyCFunction)backward, METH_NOARGS, "Backward"},
    {NULL}  /* Sentinel */
};

static PyMemberDef Value_members[] = {
   {"data", T_DOUBLE, offsetof(Value, data), 0, "Data"},
   {"grad", T_DOUBLE, offsetof(Value, grad), 0, "Gradient of the Object"},
   {"_prev", T_OBJECT, offsetof(Value, prev), 0, "Children"}, // DEBUG
   {"_op", T_OBJECT, offsetof(Value, op), 0, "Label"},
   {NULL}
  };

PyObject * pyvalue_add(PyObject *self, PyObject *other) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  value->data = ((Value *)self)->data + ((Value *)other)->data;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(2, self, other);
  value->op = PyUnicode_FromString("+");
  value->func_idx = 0;
  return (PyObject *)value;
}

PyObject * pyvalue_mul(PyObject *self, PyObject *other) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  value->data = ((Value *)self)->data * ((Value *)other)->data;
  value->grad = 0.0;
  value->prev = PyTuple_Pack(2, self, other);
  value->op = PyUnicode_FromString("*");
  value->func_idx = 1;
  return (PyObject *)value;
}

PyObject * pyvalue_pow(PyObject *self, PyObject *other, PyObject *arg) {
  Value *value = (Value *)Value_Type.tp_alloc(&Value_Type, 0);
  value->data = pow(((Value *)self)->data, PyFloat_AsDouble(other));
  value->grad = 0.0;
  value->prev = PyTuple_Pack(1, self);
  value->oa = PyFloat_AsDouble(other);
  value->op = PyUnicode_FromString("pow");
  //value->tmp = other;
  value->func_idx = 2;
  return (PyObject *)value;
}

PyObject * pyvalue_negative(PyObject *self) {
  PyObject *other = (PyObject *)Value_Type.tp_alloc(&Value_Type, 0);
  Py_INCREF(other);
  ((Value *)other)->data = -1.0;
  ((Value *)other)->grad = 0.0;
  ((Value *)other)->prev = PyTuple_New(0);
  ((Value *)other)->func_idx = -1;
  PyObject *value = Value_Type.tp_alloc(&Value_Type, 0);
  value = pyvalue_mul(self, other);
  return value;
}

PyObject * pyvalue_subtract(PyObject *self, PyObject *other) {
  return pyvalue_add(self, pyvalue_negative(other));
}

PyObject * pyvalue_truediv(PyObject *self, PyObject *other) {
  return pyvalue_mul(self, pyvalue_pow(other, PyFloat_FromDouble(-1.0), NULL));
}

PyNumberMethods Value_as_number = {
    pyvalue_add,
    pyvalue_subtract,
    pyvalue_mul,
    0,
    0,
    pyvalue_pow,
    pyvalue_negative,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    pyvalue_truediv, 
    0,
    0,
    0,
    0,
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
  PyModule_AddObject(m, "Value", (PyObject *)&Value_Type);
  return m;
}