#pragma once

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/core/object_id.hpp>

namespace goghot {

// Environment Goghot GI (engine patch 7): ticking Environment.goghot_gi_enabled on a WorldEnvironment or a
// camera's environment is all it takes. This server keeps one GoghotGI in environment mode as an internal
// child of the running scene's root (the edited scene in the editor: not saved, not shown in the dock); it
// collects the scene and its effects are registered with the RenderingServer, which runs them for every
// view whose environment enables Goghot GI. Checked once per frame (RenderingServer frame_pre_draw).
class GoghotGIServer : public godot::Object {
	GDCLASS(GoghotGIServer, godot::Object);

	godot::ObjectID node_id;
	uint64_t root_id = 0;

protected:
	static void _bind_methods() {}

public:
	void tick();
	void connect_late();
	void shutdown();
};

} // namespace goghot
