# How to Add Obstacles to the Environment Map

This guide explains how to programmatically add static obstacles, such as pylons or stop signs, to the `kal_env_map` from any other ROS node.

The `env_map_node` exposes a ROS service that allows other nodes to add objects to the semantic map. The object's position should be specified in the `stargazer` world coordinate frame.

## 1. Service Details

- **Service Name:** `/add_static_object`
- **Service Type:** `kal_env_map/AddStaticObject`

The service request takes three arguments:

- `layer` (string): The type of object to add. Currently, "pylon" and "stop_sign" are supported.
- `x` (float32): The x-coordinate of the object in the `stargazer` frame (in meters).
- `y` (float32): The y-coordinate of the object in the `stargazer` frame (in meters).

The service response contains a single boolean:

- `accepted` (bool): `True` if the object was successfully added to the map, `False` otherwise (e.g., if the object was a duplicate or out of bounds).

## 2. Python Example

Here is a complete Python example for a ROS node that adds a pylon to the map at position (x=5.0, y=3.0).

```python
#!/usr/bin/env python3
import rospy
from kal_env_map.srv import AddStaticObject, AddStaticObjectRequest

def add_pylon_example():
    # It's good practice to wait for the service to become available before trying to use it.
    rospy.wait_for_service('/add_static_object')
    try:
        # Create a handle to the service. You can reuse this handle for multiple calls.
        add_static_object = rospy.ServiceProxy('/add_static_object', AddStaticObject)

        # Create a request object
        req = AddStaticObjectRequest()
        req.layer = "pylon"
        req.x = 5.0
        req.y = 3.0

        # Call the service with the request
        resp = add_static_object(req)

        if resp.accepted:
            rospy.loginfo("Successfully added pylon to the map.")
        else:
            rospy.logwarn("Failed to add pylon. It might be a duplicate or out of bounds.")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    rospy.init_node('obstacle_adder_example')
    add_pylon_example()
```

### Key Steps in Your Node

1. **Import necessary modules:**

    ```python
    import rospy
    from kal_env_map.srv import AddStaticObject, AddStaticObjectRequest
    ```

2. **Add dependencies to your `package.xml`:**
    Your package will need to depend on `kal_env_map`.

    ```xml
    <depend>kal_env_map</depend>
    ```

3. **Update your `CMakeLists.txt`:**
    Ensure `kal_env_map` is listed as a component in `find_package`.

    ```cmake
    find_package(catkin REQUIRED COMPONENTS
      # ... other components
      kal_env_map
    )
    ```

4. **Create a service proxy and call it:**
    As shown in the example, use `rospy.ServiceProxy` to connect to the `/add_static_object` service and then call it with your desired object's information.
