#include <gdextension_interface.h>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "exposure/histogram_exposure.h"
#include "gi/gi_node.h"
#include "gi/gi_scene.h"
#include "gi/gi_server.h"
#include "gi/screen_probe_gi.h"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

using namespace godot;

static goghot::GoghotGIServer *gi_server = nullptr;

static void initialize_goghot(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_CLASS(goghot::GoghotHistogramExposure);
	GDREGISTER_CLASS(goghot::GoghotScreenProbeGI);
	GDREGISTER_CLASS(goghot::GoghotGIColorCapture);
	GDREGISTER_CLASS(goghot::GoghotGIScene);
	GDREGISTER_CLASS(goghot::GoghotGIDebug);
	GDREGISTER_CLASS(goghot::GoghotGI);
	GDREGISTER_INTERNAL_CLASS(goghot::GoghotGIServer);
	gi_server = memnew(goghot::GoghotGIServer);
	// The RenderingServer does not exist yet at this level: connected on the first main loop iteration.
	callable_mp(gi_server, &goghot::GoghotGIServer::connect_late).call_deferred();
}

static void uninitialize_goghot(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE || gi_server == nullptr) {
		return;
	}
	RenderingServer *rs = Object::cast_to<RenderingServer>(Engine::get_singleton()->get_singleton("RenderingServer"));
	if (rs && rs->is_connected("frame_pre_draw", callable_mp(gi_server, &goghot::GoghotGIServer::tick))) {
		rs->disconnect("frame_pre_draw", callable_mp(gi_server, &goghot::GoghotGIServer::tick));
	}
	gi_server->shutdown();
	memdelete(gi_server);
	gi_server = nullptr;
}

extern "C" {
GDExtensionBool GDE_EXPORT goghot_library_init(GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
	init_obj.register_initializer(initialize_goghot);
	init_obj.register_terminator(uninitialize_goghot);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
	return init_obj.init();
}
}
