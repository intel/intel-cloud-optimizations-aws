# ===============================================================================
#  Copyright 2021-2022 Intel Corporation.
# 
#  This software and the related documents are Intel copyrighted  materials,  and
#  your use of  them is  governed by the  express license  under which  they were
#  provided to you (License).  Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute,  disclose or transmit this software or
#  the related documents without Intel's prior written permission.
# 
#  This software and the related documents  are provided as  is,  with no express
#  or implied  warranties,  other  than those  that are  expressly stated  in the
#  License.
# ===============================================================================

import predictor as myapp

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = myapp.app
